module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    func.func private @zero_bf16(memref<128x128xbf16>) attributes {link_with = "mm_scalar.o"}
    func.func private @matmul_bf16(memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) attributes {link_with = "mm_scalar.o"}
    aie.objectfifo @Ai(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>> 
    aie.objectfifo @Ao(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@Ai] -> [@Ao]([] [])
    aie.objectfifo @Bi(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo @Bo(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo.link [@Bi] -> [@Bo]([] [])
    aie.objectfifo @Ci(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @Co(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo.link [@Ci] -> [@Co]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c65535 = arith.constant 65535 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c65535 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c256 step %c1_1 {
          %0 = aie.objectfifo.acquire @Ci(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @Bo(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @Ao(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @Ao(Consume, 1)
            %6 = aie.objectfifo.acquire @Ao(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @Ao(Consume, 1)
            %8 = aie.objectfifo.acquire @Ao(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @Ao(Consume, 1)
            %10 = aie.objectfifo.acquire @Ao(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @Ao(Consume, 1)
            aie.objectfifo.release @Bo(Consume, 1)
          }
          aie.objectfifo.release @Ci(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence @seq(%arg0: memref<4194304xbf16>, %arg1: memref<4194304xbf16>, %arg2: memref<4194304xbf16>) {
      %0 = aiex.dma_configure_task_for @Ai {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 0, 262144, [<size = 1, stride = 0>, <size = 4, stride = 512>, <size = 128, stride = 2048>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @Bi {
        aie.dma_bd(%arg1 : memref<4194304xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 2048>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @Co {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 2048>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
    }
  }
}
