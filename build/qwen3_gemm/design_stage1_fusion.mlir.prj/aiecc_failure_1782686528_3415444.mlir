module {
  aie.device(npu2) {
    func.func private @rms_norm_bf16_fused(memref<32x64xbf16>, memref<1024xbf16>) attributes {link_with = "rms_norm_only.o"}
    func.func private @zero_kernel_bf16(memref<128x128xbf16>) attributes {link_with = "mm_bf16_direct.o"}
    func.func private @matmul_vectorized_bf16(memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) attributes {link_with = "mm_bf16_direct.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @AL3_0(%shim_noc_tile_0_0, {%mem_tile_0_1 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>> 
    aie.objectfifo @AL1_0(%mem_tile_0_1 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_2 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]}, 1 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@AL3_0] -> [@AL1_0]([] [])
    aie.objectfifo @NL3_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @NL1_0(%mem_tile_0_1, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@NL3_0] -> [@NL1_0]([] [])
    aie.objectfifo @BL3_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo @BL1_0(%mem_tile_0_1, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo.link [@BL3_0] -> [@BL1_0]([] [])
    aie.objectfifo @CL1_0_0(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL3_0(%mem_tile_0_1 dimensionsToStream [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo.link [@CL1_0_0] -> [@CL3_0]([0] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %6 = aie.objectfifo.acquire @NL1_0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
            func.call @rms_norm_bf16_fused(%5, %7) : (memref<32x64xbf16>, memref<1024xbf16>) -> ()
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @NL1_0(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %10 = aie.objectfifo.acquire @NL1_0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
            func.call @rms_norm_bf16_fused(%9, %11) : (memref<32x64xbf16>, memref<1024xbf16>) -> ()
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @NL1_0(Consume, 1)
            %12 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %14 = aie.objectfifo.acquire @NL1_0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
            %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
            func.call @rms_norm_bf16_fused(%13, %15) : (memref<32x64xbf16>, memref<1024xbf16>) -> ()
            func.call @matmul_vectorized_bf16(%13, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @NL1_0(Consume, 1)
            %16 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            %18 = aie.objectfifo.acquire @NL1_0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
            func.call @rms_norm_bf16_fused(%17, %19) : (memref<32x64xbf16>, memref<1024xbf16>) -> ()
            func.call @matmul_vectorized_bf16(%17, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @NL1_0(Consume, 1)
            aie.objectfifo.release @BL1_0(Consume, 1)
          }
          aie.objectfifo.release @CL1_0_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3072 : i32}
    aie.runtime_sequence @seq(%arg0: memref<1048576xbf16>, %arg1: memref<1048576xbf16>, %arg2: memref<1024xbf16>, %arg3: memref<1048576xbf16>) {
      %0 = aiex.dma_configure_task_for @AL3_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @BL3_0 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @NL3_0 {
        aie.dma_bd(%arg2 : memref<1024xbf16>, 0, 0, [<size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @CL3_0 {
        aie.dma_bd(%arg3 : memref<1048576xbf16>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
    }
  }
}
