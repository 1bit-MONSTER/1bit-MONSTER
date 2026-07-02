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
    aie.objectfifo @AS0(%shim_noc_tile_0_0, {%mem_tile_0_1 dimensionsFromStream [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x512xi8>> 
    aie.objectfifo @AC0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@AS0] -> [@AC0]([] [])
    aie.objectfifo @BS0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS0] -> [@BC0]([] [])
    aie.objectfifo @CC0(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @CS0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@CC0] -> [@CS0]([] [])
    aie.objectfifo @AS1(%shim_noc_tile_0_0, {%mem_tile_1_1 dimensionsFromStream [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x512xi8>> 
    aie.objectfifo @AC1(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@AS1] -> [@AC1]([] [])
    aie.objectfifo @BS1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC1(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS1] -> [@BC1]([] [])
    aie.objectfifo @CC1(%tile_1_2, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @CS1(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@CC1] -> [@CS1]([] [])
    aie.objectfifo @AS2(%shim_noc_tile_0_0, {%mem_tile_2_1 dimensionsFromStream [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x512xi8>> 
    aie.objectfifo @AC2(%mem_tile_2_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@AS2] -> [@AC2]([] [])
    aie.objectfifo @BS2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC2(%mem_tile_2_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS2] -> [@BC2]([] [])
    aie.objectfifo @CC2(%tile_2_2, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @CS2(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@CC2] -> [@CS2]([] [])
    aie.objectfifo @AS3(%shim_noc_tile_0_0, {%mem_tile_3_1 dimensionsFromStream [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x512xi8>> 
    aie.objectfifo @AC3(%mem_tile_3_1, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@AS3] -> [@AC3]([] [])
    aie.objectfifo @BS3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC3(%mem_tile_3_1, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS3] -> [@BC3]([] [])
    aie.objectfifo @CC3(%tile_3_2, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @CS3(%mem_tile_3_1, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@CC3] -> [@CS3]([] [])
    aie.objectfifo @AS4(%shim_noc_tile_0_0, {%mem_tile_4_1 dimensionsFromStream [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x512xi8>> 
    aie.objectfifo @AC4(%mem_tile_4_1, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@AS4] -> [@AC4]([] [])
    aie.objectfifo @BS4(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC4(%mem_tile_4_1, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS4] -> [@BC4]([] [])
    aie.objectfifo @CC4(%tile_4_2, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @CS4(%mem_tile_4_1, {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@CC4] -> [@CS4]([] [])
    aie.objectfifo @AS5(%shim_noc_tile_0_0, {%mem_tile_5_1 dimensionsFromStream [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x512xi8>> 
    aie.objectfifo @AC5(%mem_tile_5_1, {%tile_5_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@AS5] -> [@AC5]([] [])
    aie.objectfifo @BS5(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC5(%mem_tile_5_1, {%tile_5_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS5] -> [@BC5]([] [])
    aie.objectfifo @CC5(%tile_5_2, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @CS5(%mem_tile_5_1, {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@CC5] -> [@CS5]([] [])
    aie.objectfifo @AS6(%shim_noc_tile_0_0, {%mem_tile_6_1 dimensionsFromStream [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x512xi8>> 
    aie.objectfifo @AC6(%mem_tile_6_1, {%tile_6_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@AS6] -> [@AC6]([] [])
    aie.objectfifo @BS6(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC6(%mem_tile_6_1, {%tile_6_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS6] -> [@BC6]([] [])
    aie.objectfifo @CC6(%tile_6_2, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @CS6(%mem_tile_6_1, {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@CC6] -> [@CS6]([] [])
    aie.objectfifo @AS7(%shim_noc_tile_0_0, {%mem_tile_7_1 dimensionsFromStream [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x512xi8>> 
    aie.objectfifo @AC7(%mem_tile_7_1, {%tile_7_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@AS7] -> [@AC7]([] [])
    aie.objectfifo @BS7(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC7(%mem_tile_7_1, {%tile_7_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS7] -> [@BC7]([] [])
    aie.objectfifo @CC7(%tile_7_2, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @CS7(%mem_tile_7_1, {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@CC7] -> [@CS7]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @CC0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %8 = aie.objectfifo.acquire @AC0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC0(Consume, 1)
          aie.objectfifo.release @BC0(Consume, 1)
        }
        aie.objectfifo.release @CC0(Produce, 1)
        %2 = aie.objectfifo.acquire @CC0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %8 = aie.objectfifo.acquire @AC0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC0(Consume, 1)
          aie.objectfifo.release @BC0(Consume, 1)
        }
        aie.objectfifo.release @CC0(Produce, 1)
        %4 = aie.objectfifo.acquire @CC0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %8 = aie.objectfifo.acquire @AC0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC0(Consume, 1)
          aie.objectfifo.release @BC0(Consume, 1)
        }
        aie.objectfifo.release @CC0(Produce, 1)
        %6 = aie.objectfifo.acquire @CC0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %8 = aie.objectfifo.acquire @AC0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC0(Consume, 1)
          aie.objectfifo.release @BC0(Consume, 1)
        }
        aie.objectfifo.release @CC0(Produce, 1)
      }
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @CC1(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %8 = aie.objectfifo.acquire @AC1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC1(Consume, 1)
          aie.objectfifo.release @BC1(Consume, 1)
        }
        aie.objectfifo.release @CC1(Produce, 1)
        %2 = aie.objectfifo.acquire @CC1(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %8 = aie.objectfifo.acquire @AC1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC1(Consume, 1)
          aie.objectfifo.release @BC1(Consume, 1)
        }
        aie.objectfifo.release @CC1(Produce, 1)
        %4 = aie.objectfifo.acquire @CC1(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %8 = aie.objectfifo.acquire @AC1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC1(Consume, 1)
          aie.objectfifo.release @BC1(Consume, 1)
        }
        aie.objectfifo.release @CC1(Produce, 1)
        %6 = aie.objectfifo.acquire @CC1(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %8 = aie.objectfifo.acquire @AC1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC1(Consume, 1)
          aie.objectfifo.release @BC1(Consume, 1)
        }
        aie.objectfifo.release @CC1(Produce, 1)
      }
      aie.end
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @CC2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %8 = aie.objectfifo.acquire @AC2(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC2(Consume, 1)
          aie.objectfifo.release @BC2(Consume, 1)
        }
        aie.objectfifo.release @CC2(Produce, 1)
        %2 = aie.objectfifo.acquire @CC2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %8 = aie.objectfifo.acquire @AC2(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC2(Consume, 1)
          aie.objectfifo.release @BC2(Consume, 1)
        }
        aie.objectfifo.release @CC2(Produce, 1)
        %4 = aie.objectfifo.acquire @CC2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %8 = aie.objectfifo.acquire @AC2(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC2(Consume, 1)
          aie.objectfifo.release @BC2(Consume, 1)
        }
        aie.objectfifo.release @CC2(Produce, 1)
        %6 = aie.objectfifo.acquire @CC2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %8 = aie.objectfifo.acquire @AC2(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC2(Consume, 1)
          aie.objectfifo.release @BC2(Consume, 1)
        }
        aie.objectfifo.release @CC2(Produce, 1)
      }
      aie.end
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @CC3(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %8 = aie.objectfifo.acquire @AC3(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC3(Consume, 1)
          aie.objectfifo.release @BC3(Consume, 1)
        }
        aie.objectfifo.release @CC3(Produce, 1)
        %2 = aie.objectfifo.acquire @CC3(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %8 = aie.objectfifo.acquire @AC3(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC3(Consume, 1)
          aie.objectfifo.release @BC3(Consume, 1)
        }
        aie.objectfifo.release @CC3(Produce, 1)
        %4 = aie.objectfifo.acquire @CC3(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %8 = aie.objectfifo.acquire @AC3(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC3(Consume, 1)
          aie.objectfifo.release @BC3(Consume, 1)
        }
        aie.objectfifo.release @CC3(Produce, 1)
        %6 = aie.objectfifo.acquire @CC3(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %8 = aie.objectfifo.acquire @AC3(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC3(Consume, 1)
          aie.objectfifo.release @BC3(Consume, 1)
        }
        aie.objectfifo.release @CC3(Produce, 1)
      }
      aie.end
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @CC4(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %8 = aie.objectfifo.acquire @AC4(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC4(Consume, 1)
          aie.objectfifo.release @BC4(Consume, 1)
        }
        aie.objectfifo.release @CC4(Produce, 1)
        %2 = aie.objectfifo.acquire @CC4(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %8 = aie.objectfifo.acquire @AC4(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC4(Consume, 1)
          aie.objectfifo.release @BC4(Consume, 1)
        }
        aie.objectfifo.release @CC4(Produce, 1)
        %4 = aie.objectfifo.acquire @CC4(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %8 = aie.objectfifo.acquire @AC4(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC4(Consume, 1)
          aie.objectfifo.release @BC4(Consume, 1)
        }
        aie.objectfifo.release @CC4(Produce, 1)
        %6 = aie.objectfifo.acquire @CC4(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %8 = aie.objectfifo.acquire @AC4(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC4(Consume, 1)
          aie.objectfifo.release @BC4(Consume, 1)
        }
        aie.objectfifo.release @CC4(Produce, 1)
      }
      aie.end
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @CC5(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %8 = aie.objectfifo.acquire @AC5(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC5(Consume, 1)
          aie.objectfifo.release @BC5(Consume, 1)
        }
        aie.objectfifo.release @CC5(Produce, 1)
        %2 = aie.objectfifo.acquire @CC5(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %8 = aie.objectfifo.acquire @AC5(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC5(Consume, 1)
          aie.objectfifo.release @BC5(Consume, 1)
        }
        aie.objectfifo.release @CC5(Produce, 1)
        %4 = aie.objectfifo.acquire @CC5(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %8 = aie.objectfifo.acquire @AC5(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC5(Consume, 1)
          aie.objectfifo.release @BC5(Consume, 1)
        }
        aie.objectfifo.release @CC5(Produce, 1)
        %6 = aie.objectfifo.acquire @CC5(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %8 = aie.objectfifo.acquire @AC5(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC5(Consume, 1)
          aie.objectfifo.release @BC5(Consume, 1)
        }
        aie.objectfifo.release @CC5(Produce, 1)
      }
      aie.end
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @CC6(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %8 = aie.objectfifo.acquire @AC6(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC6(Consume, 1)
          aie.objectfifo.release @BC6(Consume, 1)
        }
        aie.objectfifo.release @CC6(Produce, 1)
        %2 = aie.objectfifo.acquire @CC6(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %8 = aie.objectfifo.acquire @AC6(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC6(Consume, 1)
          aie.objectfifo.release @BC6(Consume, 1)
        }
        aie.objectfifo.release @CC6(Produce, 1)
        %4 = aie.objectfifo.acquire @CC6(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %8 = aie.objectfifo.acquire @AC6(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC6(Consume, 1)
          aie.objectfifo.release @BC6(Consume, 1)
        }
        aie.objectfifo.release @CC6(Produce, 1)
        %6 = aie.objectfifo.acquire @CC6(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %8 = aie.objectfifo.acquire @AC6(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC6(Consume, 1)
          aie.objectfifo.release @BC6(Consume, 1)
        }
        aie.objectfifo.release @CC6(Produce, 1)
      }
      aie.end
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @CC7(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %8 = aie.objectfifo.acquire @AC7(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC7(Consume, 1)
          aie.objectfifo.release @BC7(Consume, 1)
        }
        aie.objectfifo.release @CC7(Produce, 1)
        %2 = aie.objectfifo.acquire @CC7(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %8 = aie.objectfifo.acquire @AC7(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC7(Consume, 1)
          aie.objectfifo.release @BC7(Consume, 1)
        }
        aie.objectfifo.release @CC7(Produce, 1)
        %4 = aie.objectfifo.acquire @CC7(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %8 = aie.objectfifo.acquire @AC7(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC7(Consume, 1)
          aie.objectfifo.release @BC7(Consume, 1)
        }
        aie.objectfifo.release @CC7(Produce, 1)
        %6 = aie.objectfifo.acquire @CC7(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %8 = aie.objectfifo.acquire @AC7(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %10 = aie.objectfifo.acquire @BC7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%9, %11, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @AC7(Consume, 1)
          aie.objectfifo.release @BC7(Consume, 1)
        }
        aie.objectfifo.release @CC7(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence @seq(%arg0: memref<131072xi8>, %arg1: memref<1048576xi8>, %arg2: memref<131072xi16>) {
      %0 = aiex.dma_configure_task_for @AS0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @BS0 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @AS1 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @BS1 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @AS2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @BS2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @AS3 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @BS3 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @AS4 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @BS4 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @AS5 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @BS5 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @AS6 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @BS6 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @AS7 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @BS7 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @CS0 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @CS1 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 128, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @CS2 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 256, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @CS3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 384, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @CS4 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 512, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @CS5 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 640, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @CS6 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @CS7 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @AS0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @BS0 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @AS1 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @BS1 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @AS2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @BS2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @AS3 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @BS3 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @AS4 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @BS4 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @AS5 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @BS5 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @AS6 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @BS6 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @AS7 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @BS7 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @CS0 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 32768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @CS1 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 32896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @CS2 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33024, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @CS3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33152, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @CS4 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33280, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @CS5 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33408, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @CS6 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @CS7 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @AS0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @BS0 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @AS1 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @BS1 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @AS2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @BS2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @AS3 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @BS3 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @AS4 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @BS4 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @AS5 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @BS5 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @AS6 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @BS6 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @AS7 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @BS7 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @CS0 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @CS1 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @CS2 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65792, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @CS3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65920, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @CS4 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66048, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @CS5 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66176, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @CS6 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @CS7 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @AS0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @BS0 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @AS1 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @BS1 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @AS2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @BS2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @AS3 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @BS3 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @AS4 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @BS4 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @AS5 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @BS5 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @AS6 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @BS6 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @AS7 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @BS7 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @CS0 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @CS1 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @CS2 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98560, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @CS3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98688, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @CS4 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98816, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @CS5 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98944, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @CS6 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 99072, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @CS7 {
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
  }
}

