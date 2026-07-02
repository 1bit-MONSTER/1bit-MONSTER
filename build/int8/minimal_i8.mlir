module {
  aie.device(npu2) {
    func.func private @matmul_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
    func.func private @zero_i16(memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @a_s(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo @a_c(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@a_s] -> [@a_c]([] [])
    aie.objectfifo @b_s(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @b_c(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@b_s] -> [@b_c]([] [])
    aie.objectfifo @c_c(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @c_s(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@c_c] -> [@c_s]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @c_c(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
      func.call @zero_i16(%1) : (memref<32x128xi16>) -> ()
      %2 = aie.objectfifo.acquire @a_c(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
      %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
      %4 = aie.objectfifo.acquire @b_c(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
      %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
      func.call @matmul_i8_i16(%3, %5, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.objectfifo.release @a_c(Consume, 1)
      aie.objectfifo.release @b_c(Consume, 1)
      aie.objectfifo.release @c_c(Produce, 1)
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<2048xi8>, %arg1: memref<8192xi8>, %arg2: memref<4096xi16>) {
      %0 = aiex.dma_configure_task_for @a_s {
        aie.dma_bd(%arg0 : memref<2048xi8>, 0, 0, [<size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %1 = aiex.dma_configure_task_for @b_s {
        aie.dma_bd(%arg1 : memref<8192xi8>, 0, 0, [<size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %2 = aiex.dma_configure_task_for @c_s {
        aie.dma_bd(%arg2 : memref<4096xi16>, 0, 0, [<size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
    }
  }
}

