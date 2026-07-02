module {
  aie.device(npu2) {
    func.func private @zero_kernel_bf16(memref<128x128xi16>) attributes {link_with = "mm_128x64x128.o"}
    func.func private @matmul_scalar_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) attributes {link_with = "mm_128x64x128.o"}
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
    aie.objectfifo @A_L3L2(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo @A_L2L1(%mem_tile_0_1, {%tile_0_2, %tile_1_2, %tile_2_2, %tile_3_2, %tile_4_2, %tile_5_2, %tile_6_2, %tile_7_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_L3L2] -> [@A_L2L1]([] [])
    aie.objectfifo.link [@A_L3L2] -> [@A_L2L1]([] [])
    aie.objectfifo @BS0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS0] -> [@BC0]([] [])
    aie.objectfifo @CC0(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @CS0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@CC0] -> [@CS0]([] [])
    aie.objectfifo @BS1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC1(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS1] -> [@BC1]([] [])
    aie.objectfifo @CC1(%tile_1_2, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @CS1(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@CC1] -> [@CS1]([] [])
    aie.objectfifo @BS2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC2(%mem_tile_2_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS2] -> [@BC2]([] [])
    aie.objectfifo @CC2(%tile_2_2, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @CS2(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@CC2] -> [@CS2]([] [])
    aie.objectfifo @BS3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC3(%mem_tile_3_1, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS3] -> [@BC3]([] [])
    aie.objectfifo @CC3(%tile_3_2, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @CS3(%mem_tile_3_1, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@CC3] -> [@CS3]([] [])
    aie.objectfifo @BS4(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC4(%mem_tile_4_1, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS4] -> [@BC4]([] [])
    aie.objectfifo @CC4(%tile_4_2, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @CS4(%mem_tile_4_1, {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@CC4] -> [@CS4]([] [])
    aie.objectfifo @BS5(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC5(%mem_tile_5_1, {%tile_5_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS5] -> [@BC5]([] [])
    aie.objectfifo @CC5(%tile_5_2, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @CS5(%mem_tile_5_1, {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@CC5] -> [@CS5]([] [])
    aie.objectfifo @BS6(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC6(%mem_tile_6_1, {%tile_6_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS6] -> [@BC6]([] [])
    aie.objectfifo @CC6(%tile_6_2, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @CS6(%mem_tile_6_1, {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@CC6] -> [@CS6]([] [])
    aie.objectfifo @BS7(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @BC7(%mem_tile_7_1, {%tile_7_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@BS7] -> [@BC7]([] [])
    aie.objectfifo @CC7(%tile_7_2, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @CS7(%mem_tile_7_1, {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@CC7] -> [@CS7]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @CC0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %2 = aie.objectfifo.acquire @BC0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %4 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
            func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
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
        %0 = aie.objectfifo.acquire @CC1(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %2 = aie.objectfifo.acquire @BC1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %4 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
            func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
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
        %0 = aie.objectfifo.acquire @CC2(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %2 = aie.objectfifo.acquire @BC2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %4 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
            func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
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
        %0 = aie.objectfifo.acquire @CC3(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %2 = aie.objectfifo.acquire @BC3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %4 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
            func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
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
        %0 = aie.objectfifo.acquire @CC4(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %2 = aie.objectfifo.acquire @BC4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %4 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
            func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
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
        %0 = aie.objectfifo.acquire @CC5(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %2 = aie.objectfifo.acquire @BC5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %4 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
            func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
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
        %0 = aie.objectfifo.acquire @CC6(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %2 = aie.objectfifo.acquire @BC6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %4 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
            func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
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
        %0 = aie.objectfifo.acquire @CC7(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %2 = aie.objectfifo.acquire @BC7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %4 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
            func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
            aie.objectfifo.release @A_L2L1(Consume, 1)
          }
          aie.objectfifo.release @BC7(Consume, 1)
        }
        aie.objectfifo.release @CC7(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence @seq(%arg0: memref<131072xi8>, %arg1: memref<1048576xi8>, %arg2: memref<131072xi16>) {
      %0 = aiex.dma_configure_task_for @BS0 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @BS1 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @BS2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @BS3 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @BS4 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @BS5 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @BS6 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @BS7 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @CS0 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @CS1 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 128, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @CS2 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 256, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @CS3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @CS4 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 512, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @CS5 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 640, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @CS6 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @CS7 {
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
  }
}
