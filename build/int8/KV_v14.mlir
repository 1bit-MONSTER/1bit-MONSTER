module {
  aie.device(npu2) {
    func.func private @zero_scalar_i16(memref<128x128xi16>) attributes {link_with = "mm_128x64x128.o"}
    func.func private @matmul_scalar_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) attributes {link_with = "mm_128x64x128.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    aie.objectfifo @A_L3L2(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<128x512xi8>> 
    aie.objectfifo @A_L2L1(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_L3L2] -> [@A_L2L1]([] [])
    aie.objectfifo @B_L3L2(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @B_L2L1(%mem_tile_1_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@B_L3L2] -> [@B_L2L1]([] [])
    aie.objectfifo @C_L1L2(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @C_L2L3(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@C_L1L2] -> [@C_L2L3]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %18 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%17, %19, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<128x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %18 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%17, %19, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<128x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %18 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%17, %19, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<128x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %18 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%17, %19, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<128x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %18 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%17, %19, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<128x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %18 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%17, %19, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%13) : (memref<128x128xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c16_18 = arith.constant 16 : index
        %c1_19 = arith.constant 1 : index
        scf.for %arg1 = %c0_17 to %c16_18 step %c1_19 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %18 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%17, %19, %13) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%15) : (memref<128x128xi16>) -> ()
        %c0_20 = arith.constant 0 : index
        %c16_21 = arith.constant 16 : index
        %c1_22 = arith.constant 1 : index
        scf.for %arg1 = %c0_20 to %c16_21 step %c1_22 {
          %16 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %18 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%17, %19, %15) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence @seq(%arg0: memref<131072xi8>, %arg1: memref<1048576xi8>, %arg2: memref<131072xi16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 128, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 256, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 512, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 640, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 896, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%23)
    }
  }
}

