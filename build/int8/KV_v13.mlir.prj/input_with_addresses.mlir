module {
  aie.device(npu2) {
    func.func private @zero_scalar_i16(memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
    func.func private @matmul_scalar_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %C_L2L3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L2L3_cons_prod_lock_0"}
    %C_L2L3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L2L3_cons_cons_lock_0"}
    %C_L1L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 32768 : i32, sym_name = "C_L1L2_cons_buff_0"} : memref<32x128xi16> 
    %C_L1L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32, sym_name = "C_L1L2_cons_prod_lock_0"}
    %C_L1L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "C_L1L2_cons_cons_lock_0"}
    %C_L1L2_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "C_L1L2_buff_0"} : memref<32x128xi16> 
    %C_L1L2_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "C_L1L2_prod_lock_0"}
    %C_L1L2_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_L1L2_cons_lock_0"}
    %B_L2L1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 9216 : i32, sym_name = "B_L2L1_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 17408 : i32, sym_name = "B_L2L1_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "B_L2L1_cons_prod_lock_0"}
    %B_L2L1_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_L2L1_cons_cons_lock_0"}
    %B_L3L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 40960 : i32, sym_name = "B_L3L2_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 49152 : i32, sym_name = "B_L3L2_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "B_L3L2_cons_prod_lock_0"}
    %B_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "B_L3L2_cons_cons_lock_0"}
    %B_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_L3L2_prod_lock_0"}
    %B_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_L3L2_cons_lock_0"}
    %A_L2L1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 25600 : i32, sym_name = "A_L2L1_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 27648 : i32, sym_name = "A_L2L1_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L2L1_cons_prod_lock_0"}
    %A_L2L1_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L2L1_cons_cons_lock_0"}
    %A_L3L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "A_L3L2_cons_buff_0"} : memref<32x512xi8> 
    %A_L3L2_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 16384 : i32, sym_name = "A_L3L2_cons_buff_1"} : memref<32x512xi8> 
    %A_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "A_L3L2_cons_prod_lock_0"}
    %A_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "A_L3L2_cons_cons_lock_0"}
    %A_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L2_prod_lock_0"}
    %A_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L2_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 29696 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb354
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb355
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
    ^bb4:  // pred: ^bb3
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb8(%7: memref<32x64xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb12(%11: memref<64x128xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_scalar_i8_i16(%7, %11, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous0[%c2] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb19(%32: memref<32x64xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb23(%36: memref<64x128xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_scalar_i8_i16(%32, %36, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous0[%c2] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb30(%57: memref<32x64xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb34(%61: memref<64x128xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_scalar_i8_i16(%57, %61, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous0[%c2] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb41(%82: memref<32x64xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb45(%86: memref<64x128xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_scalar_i8_i16(%82, %86, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous0[%c2] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb47(%c0 : index)
    ^bb47(%102: index):  // 2 preds: ^bb46, ^bb56
      %103 = arith.cmpi slt, %102, %c16 : index
      cf.cond_br %103, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %104 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %105 = arith.index_cast %104 : i32 to index
      %106 = arith.index_cast %105 : index to i32
      cf.switch %106 : i32, [
        default: ^bb51,
        0: ^bb49,
        1: ^bb50
      ]
    ^bb49:  // pred: ^bb48
      cf.br ^bb52(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb50:  // pred: ^bb48
      cf.br ^bb52(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb51:  // pred: ^bb48
      cf.br ^bb52(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb52(%107: memref<32x64xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb55,
        0: ^bb53,
        1: ^bb54
      ]
    ^bb53:  // pred: ^bb52
      cf.br ^bb56(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb54:  // pred: ^bb52
      cf.br ^bb56(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb55:  // pred: ^bb52
      cf.br ^bb56(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb56(%111: memref<64x128xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_scalar_i8_i16(%107, %111, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %112 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %113 = arith.addi %112, %c1_i32 : i32
      %114 = arith.cmpi sge, %113, %c2_i32 : i32
      %115 = arith.subi %113, %c2_i32 : i32
      %116 = arith.select %114, %115, %113 : i32
      memref.store %116, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %117 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %118 = arith.addi %117, %c1_i32 : i32
      %119 = arith.cmpi sge, %118, %c2_i32 : i32
      %120 = arith.subi %118, %c2_i32 : i32
      %121 = arith.select %119, %120, %118 : i32
      memref.store %121, %_anonymous0[%c2] : memref<3xi32>
      %122 = arith.addi %102, %c1 : index
      cf.br ^bb47(%122 : index)
    ^bb57:  // pred: ^bb47
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %123 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c1_i32 : i32
      %126 = arith.select %125, %123, %124 : i32
      memref.store %126, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb58(%c0 : index)
    ^bb58(%127: index):  // 2 preds: ^bb57, ^bb67
      %128 = arith.cmpi slt, %127, %c16 : index
      cf.cond_br %128, ^bb59, ^bb68
    ^bb59:  // pred: ^bb58
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %129 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %130 = arith.index_cast %129 : i32 to index
      %131 = arith.index_cast %130 : index to i32
      cf.switch %131 : i32, [
        default: ^bb62,
        0: ^bb60,
        1: ^bb61
      ]
    ^bb60:  // pred: ^bb59
      cf.br ^bb63(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb61:  // pred: ^bb59
      cf.br ^bb63(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb62:  // pred: ^bb59
      cf.br ^bb63(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb63(%132: memref<32x64xi8>):  // 3 preds: ^bb60, ^bb61, ^bb62
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %133 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %134 = arith.index_cast %133 : i32 to index
      %135 = arith.index_cast %134 : index to i32
      cf.switch %135 : i32, [
        default: ^bb66,
        0: ^bb64,
        1: ^bb65
      ]
    ^bb64:  // pred: ^bb63
      cf.br ^bb67(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb65:  // pred: ^bb63
      cf.br ^bb67(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb66:  // pred: ^bb63
      cf.br ^bb67(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb67(%136: memref<64x128xi8>):  // 3 preds: ^bb64, ^bb65, ^bb66
      func.call @matmul_scalar_i8_i16(%132, %136, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %137 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %138 = arith.addi %137, %c1_i32 : i32
      %139 = arith.cmpi sge, %138, %c2_i32 : i32
      %140 = arith.subi %138, %c2_i32 : i32
      %141 = arith.select %139, %140, %138 : i32
      memref.store %141, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %142 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %143 = arith.addi %142, %c1_i32 : i32
      %144 = arith.cmpi sge, %143, %c2_i32 : i32
      %145 = arith.subi %143, %c2_i32 : i32
      %146 = arith.select %144, %145, %143 : i32
      memref.store %146, %_anonymous0[%c2] : memref<3xi32>
      %147 = arith.addi %127, %c1 : index
      cf.br ^bb58(%147 : index)
    ^bb68:  // pred: ^bb58
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %148 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c1_i32 : i32
      %151 = arith.select %150, %148, %149 : i32
      memref.store %151, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb69(%c0 : index)
    ^bb69(%152: index):  // 2 preds: ^bb68, ^bb78
      %153 = arith.cmpi slt, %152, %c16 : index
      cf.cond_br %153, ^bb70, ^bb79
    ^bb70:  // pred: ^bb69
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %154 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %155 = arith.index_cast %154 : i32 to index
      %156 = arith.index_cast %155 : index to i32
      cf.switch %156 : i32, [
        default: ^bb73,
        0: ^bb71,
        1: ^bb72
      ]
    ^bb71:  // pred: ^bb70
      cf.br ^bb74(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb72:  // pred: ^bb70
      cf.br ^bb74(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb73:  // pred: ^bb70
      cf.br ^bb74(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb74(%157: memref<32x64xi8>):  // 3 preds: ^bb71, ^bb72, ^bb73
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %158 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %159 = arith.index_cast %158 : i32 to index
      %160 = arith.index_cast %159 : index to i32
      cf.switch %160 : i32, [
        default: ^bb77,
        0: ^bb75,
        1: ^bb76
      ]
    ^bb75:  // pred: ^bb74
      cf.br ^bb78(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb76:  // pred: ^bb74
      cf.br ^bb78(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb77:  // pred: ^bb74
      cf.br ^bb78(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb78(%161: memref<64x128xi8>):  // 3 preds: ^bb75, ^bb76, ^bb77
      func.call @matmul_scalar_i8_i16(%157, %161, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %162 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %163 = arith.addi %162, %c1_i32 : i32
      %164 = arith.cmpi sge, %163, %c2_i32 : i32
      %165 = arith.subi %163, %c2_i32 : i32
      %166 = arith.select %164, %165, %163 : i32
      memref.store %166, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %167 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %168 = arith.addi %167, %c1_i32 : i32
      %169 = arith.cmpi sge, %168, %c2_i32 : i32
      %170 = arith.subi %168, %c2_i32 : i32
      %171 = arith.select %169, %170, %168 : i32
      memref.store %171, %_anonymous0[%c2] : memref<3xi32>
      %172 = arith.addi %152, %c1 : index
      cf.br ^bb69(%172 : index)
    ^bb79:  // pred: ^bb69
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %173 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %174 = arith.addi %173, %c1_i32 : i32
      %175 = arith.cmpi sge, %174, %c1_i32 : i32
      %176 = arith.select %175, %173, %174 : i32
      memref.store %176, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb80(%c0 : index)
    ^bb80(%177: index):  // 2 preds: ^bb79, ^bb89
      %178 = arith.cmpi slt, %177, %c16 : index
      cf.cond_br %178, ^bb81, ^bb90
    ^bb81:  // pred: ^bb80
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %179 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %180 = arith.index_cast %179 : i32 to index
      %181 = arith.index_cast %180 : index to i32
      cf.switch %181 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb85(%182: memref<32x64xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %183 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %184 = arith.index_cast %183 : i32 to index
      %185 = arith.index_cast %184 : index to i32
      cf.switch %185 : i32, [
        default: ^bb88,
        0: ^bb86,
        1: ^bb87
      ]
    ^bb86:  // pred: ^bb85
      cf.br ^bb89(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb87:  // pred: ^bb85
      cf.br ^bb89(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb88:  // pred: ^bb85
      cf.br ^bb89(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb89(%186: memref<64x128xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_scalar_i8_i16(%182, %186, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %187 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %188 = arith.addi %187, %c1_i32 : i32
      %189 = arith.cmpi sge, %188, %c2_i32 : i32
      %190 = arith.subi %188, %c2_i32 : i32
      %191 = arith.select %189, %190, %188 : i32
      memref.store %191, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %192 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %193 = arith.addi %192, %c1_i32 : i32
      %194 = arith.cmpi sge, %193, %c2_i32 : i32
      %195 = arith.subi %193, %c2_i32 : i32
      %196 = arith.select %194, %195, %193 : i32
      memref.store %196, %_anonymous0[%c2] : memref<3xi32>
      %197 = arith.addi %177, %c1 : index
      cf.br ^bb80(%197 : index)
    ^bb90:  // pred: ^bb80
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %198 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %199 = arith.addi %198, %c1_i32 : i32
      %200 = arith.cmpi sge, %199, %c1_i32 : i32
      %201 = arith.select %200, %198, %199 : i32
      memref.store %201, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb91(%c0 : index)
    ^bb91(%202: index):  // 2 preds: ^bb90, ^bb100
      %203 = arith.cmpi slt, %202, %c16 : index
      cf.cond_br %203, ^bb92, ^bb101
    ^bb92:  // pred: ^bb91
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %204 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %205 = arith.index_cast %204 : i32 to index
      %206 = arith.index_cast %205 : index to i32
      cf.switch %206 : i32, [
        default: ^bb95,
        0: ^bb93,
        1: ^bb94
      ]
    ^bb93:  // pred: ^bb92
      cf.br ^bb96(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb94:  // pred: ^bb92
      cf.br ^bb96(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb95:  // pred: ^bb92
      cf.br ^bb96(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb96(%207: memref<32x64xi8>):  // 3 preds: ^bb93, ^bb94, ^bb95
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %208 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %209 = arith.index_cast %208 : i32 to index
      %210 = arith.index_cast %209 : index to i32
      cf.switch %210 : i32, [
        default: ^bb99,
        0: ^bb97,
        1: ^bb98
      ]
    ^bb97:  // pred: ^bb96
      cf.br ^bb100(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb98:  // pred: ^bb96
      cf.br ^bb100(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb99:  // pred: ^bb96
      cf.br ^bb100(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb100(%211: memref<64x128xi8>):  // 3 preds: ^bb97, ^bb98, ^bb99
      func.call @matmul_scalar_i8_i16(%207, %211, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %212 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %213 = arith.addi %212, %c1_i32 : i32
      %214 = arith.cmpi sge, %213, %c2_i32 : i32
      %215 = arith.subi %213, %c2_i32 : i32
      %216 = arith.select %214, %215, %213 : i32
      memref.store %216, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %217 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %218 = arith.addi %217, %c1_i32 : i32
      %219 = arith.cmpi sge, %218, %c2_i32 : i32
      %220 = arith.subi %218, %c2_i32 : i32
      %221 = arith.select %219, %220, %218 : i32
      memref.store %221, %_anonymous0[%c2] : memref<3xi32>
      %222 = arith.addi %202, %c1 : index
      cf.br ^bb91(%222 : index)
    ^bb101:  // pred: ^bb91
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %223 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %224 = arith.addi %223, %c1_i32 : i32
      %225 = arith.cmpi sge, %224, %c1_i32 : i32
      %226 = arith.select %225, %223, %224 : i32
      memref.store %226, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb102(%c0 : index)
    ^bb102(%227: index):  // 2 preds: ^bb101, ^bb111
      %228 = arith.cmpi slt, %227, %c16 : index
      cf.cond_br %228, ^bb103, ^bb112
    ^bb103:  // pred: ^bb102
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %229 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %230 = arith.index_cast %229 : i32 to index
      %231 = arith.index_cast %230 : index to i32
      cf.switch %231 : i32, [
        default: ^bb106,
        0: ^bb104,
        1: ^bb105
      ]
    ^bb104:  // pred: ^bb103
      cf.br ^bb107(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb105:  // pred: ^bb103
      cf.br ^bb107(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb106:  // pred: ^bb103
      cf.br ^bb107(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb107(%232: memref<32x64xi8>):  // 3 preds: ^bb104, ^bb105, ^bb106
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %233 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %234 = arith.index_cast %233 : i32 to index
      %235 = arith.index_cast %234 : index to i32
      cf.switch %235 : i32, [
        default: ^bb110,
        0: ^bb108,
        1: ^bb109
      ]
    ^bb108:  // pred: ^bb107
      cf.br ^bb111(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb109:  // pred: ^bb107
      cf.br ^bb111(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb110:  // pred: ^bb107
      cf.br ^bb111(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb111(%236: memref<64x128xi8>):  // 3 preds: ^bb108, ^bb109, ^bb110
      func.call @matmul_scalar_i8_i16(%232, %236, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %237 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %238 = arith.addi %237, %c1_i32 : i32
      %239 = arith.cmpi sge, %238, %c2_i32 : i32
      %240 = arith.subi %238, %c2_i32 : i32
      %241 = arith.select %239, %240, %238 : i32
      memref.store %241, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %242 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %243 = arith.addi %242, %c1_i32 : i32
      %244 = arith.cmpi sge, %243, %c2_i32 : i32
      %245 = arith.subi %243, %c2_i32 : i32
      %246 = arith.select %244, %245, %243 : i32
      memref.store %246, %_anonymous0[%c2] : memref<3xi32>
      %247 = arith.addi %227, %c1 : index
      cf.br ^bb102(%247 : index)
    ^bb112:  // pred: ^bb102
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %248 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %249 = arith.addi %248, %c1_i32 : i32
      %250 = arith.cmpi sge, %249, %c1_i32 : i32
      %251 = arith.select %250, %248, %249 : i32
      memref.store %251, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb113(%c0 : index)
    ^bb113(%252: index):  // 2 preds: ^bb112, ^bb122
      %253 = arith.cmpi slt, %252, %c16 : index
      cf.cond_br %253, ^bb114, ^bb123
    ^bb114:  // pred: ^bb113
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %254 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %255 = arith.index_cast %254 : i32 to index
      %256 = arith.index_cast %255 : index to i32
      cf.switch %256 : i32, [
        default: ^bb117,
        0: ^bb115,
        1: ^bb116
      ]
    ^bb115:  // pred: ^bb114
      cf.br ^bb118(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb116:  // pred: ^bb114
      cf.br ^bb118(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb117:  // pred: ^bb114
      cf.br ^bb118(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb118(%257: memref<32x64xi8>):  // 3 preds: ^bb115, ^bb116, ^bb117
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %258 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %259 = arith.index_cast %258 : i32 to index
      %260 = arith.index_cast %259 : index to i32
      cf.switch %260 : i32, [
        default: ^bb121,
        0: ^bb119,
        1: ^bb120
      ]
    ^bb119:  // pred: ^bb118
      cf.br ^bb122(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb120:  // pred: ^bb118
      cf.br ^bb122(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb121:  // pred: ^bb118
      cf.br ^bb122(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb122(%261: memref<64x128xi8>):  // 3 preds: ^bb119, ^bb120, ^bb121
      func.call @matmul_scalar_i8_i16(%257, %261, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %262 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %263 = arith.addi %262, %c1_i32 : i32
      %264 = arith.cmpi sge, %263, %c2_i32 : i32
      %265 = arith.subi %263, %c2_i32 : i32
      %266 = arith.select %264, %265, %263 : i32
      memref.store %266, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %267 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %268 = arith.addi %267, %c1_i32 : i32
      %269 = arith.cmpi sge, %268, %c2_i32 : i32
      %270 = arith.subi %268, %c2_i32 : i32
      %271 = arith.select %269, %270, %268 : i32
      memref.store %271, %_anonymous0[%c2] : memref<3xi32>
      %272 = arith.addi %252, %c1 : index
      cf.br ^bb113(%272 : index)
    ^bb123:  // pred: ^bb113
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %273 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %274 = arith.addi %273, %c1_i32 : i32
      %275 = arith.cmpi sge, %274, %c1_i32 : i32
      %276 = arith.select %275, %273, %274 : i32
      memref.store %276, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb124(%c0 : index)
    ^bb124(%277: index):  // 2 preds: ^bb123, ^bb133
      %278 = arith.cmpi slt, %277, %c16 : index
      cf.cond_br %278, ^bb125, ^bb134
    ^bb125:  // pred: ^bb124
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %279 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %280 = arith.index_cast %279 : i32 to index
      %281 = arith.index_cast %280 : index to i32
      cf.switch %281 : i32, [
        default: ^bb128,
        0: ^bb126,
        1: ^bb127
      ]
    ^bb126:  // pred: ^bb125
      cf.br ^bb129(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb127:  // pred: ^bb125
      cf.br ^bb129(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb128:  // pred: ^bb125
      cf.br ^bb129(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb129(%282: memref<32x64xi8>):  // 3 preds: ^bb126, ^bb127, ^bb128
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %283 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %284 = arith.index_cast %283 : i32 to index
      %285 = arith.index_cast %284 : index to i32
      cf.switch %285 : i32, [
        default: ^bb132,
        0: ^bb130,
        1: ^bb131
      ]
    ^bb130:  // pred: ^bb129
      cf.br ^bb133(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb131:  // pred: ^bb129
      cf.br ^bb133(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb132:  // pred: ^bb129
      cf.br ^bb133(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb133(%286: memref<64x128xi8>):  // 3 preds: ^bb130, ^bb131, ^bb132
      func.call @matmul_scalar_i8_i16(%282, %286, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %287 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %288 = arith.addi %287, %c1_i32 : i32
      %289 = arith.cmpi sge, %288, %c2_i32 : i32
      %290 = arith.subi %288, %c2_i32 : i32
      %291 = arith.select %289, %290, %288 : i32
      memref.store %291, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %292 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %293 = arith.addi %292, %c1_i32 : i32
      %294 = arith.cmpi sge, %293, %c2_i32 : i32
      %295 = arith.subi %293, %c2_i32 : i32
      %296 = arith.select %294, %295, %293 : i32
      memref.store %296, %_anonymous0[%c2] : memref<3xi32>
      %297 = arith.addi %277, %c1 : index
      cf.br ^bb124(%297 : index)
    ^bb134:  // pred: ^bb124
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %298 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %299 = arith.addi %298, %c1_i32 : i32
      %300 = arith.cmpi sge, %299, %c1_i32 : i32
      %301 = arith.select %300, %298, %299 : i32
      memref.store %301, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb135(%c0 : index)
    ^bb135(%302: index):  // 2 preds: ^bb134, ^bb144
      %303 = arith.cmpi slt, %302, %c16 : index
      cf.cond_br %303, ^bb136, ^bb145
    ^bb136:  // pred: ^bb135
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %304 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %305 = arith.index_cast %304 : i32 to index
      %306 = arith.index_cast %305 : index to i32
      cf.switch %306 : i32, [
        default: ^bb139,
        0: ^bb137,
        1: ^bb138
      ]
    ^bb137:  // pred: ^bb136
      cf.br ^bb140(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb138:  // pred: ^bb136
      cf.br ^bb140(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb139:  // pred: ^bb136
      cf.br ^bb140(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb140(%307: memref<32x64xi8>):  // 3 preds: ^bb137, ^bb138, ^bb139
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %308 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %309 = arith.index_cast %308 : i32 to index
      %310 = arith.index_cast %309 : index to i32
      cf.switch %310 : i32, [
        default: ^bb143,
        0: ^bb141,
        1: ^bb142
      ]
    ^bb141:  // pred: ^bb140
      cf.br ^bb144(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb142:  // pred: ^bb140
      cf.br ^bb144(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb143:  // pred: ^bb140
      cf.br ^bb144(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb144(%311: memref<64x128xi8>):  // 3 preds: ^bb141, ^bb142, ^bb143
      func.call @matmul_scalar_i8_i16(%307, %311, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %312 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %313 = arith.addi %312, %c1_i32 : i32
      %314 = arith.cmpi sge, %313, %c2_i32 : i32
      %315 = arith.subi %313, %c2_i32 : i32
      %316 = arith.select %314, %315, %313 : i32
      memref.store %316, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %317 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %318 = arith.addi %317, %c1_i32 : i32
      %319 = arith.cmpi sge, %318, %c2_i32 : i32
      %320 = arith.subi %318, %c2_i32 : i32
      %321 = arith.select %319, %320, %318 : i32
      memref.store %321, %_anonymous0[%c2] : memref<3xi32>
      %322 = arith.addi %302, %c1 : index
      cf.br ^bb135(%322 : index)
    ^bb145:  // pred: ^bb135
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %323 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %324 = arith.addi %323, %c1_i32 : i32
      %325 = arith.cmpi sge, %324, %c1_i32 : i32
      %326 = arith.select %325, %323, %324 : i32
      memref.store %326, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb146(%c0 : index)
    ^bb146(%327: index):  // 2 preds: ^bb145, ^bb155
      %328 = arith.cmpi slt, %327, %c16 : index
      cf.cond_br %328, ^bb147, ^bb156
    ^bb147:  // pred: ^bb146
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %329 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %330 = arith.index_cast %329 : i32 to index
      %331 = arith.index_cast %330 : index to i32
      cf.switch %331 : i32, [
        default: ^bb150,
        0: ^bb148,
        1: ^bb149
      ]
    ^bb148:  // pred: ^bb147
      cf.br ^bb151(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb149:  // pred: ^bb147
      cf.br ^bb151(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb150:  // pred: ^bb147
      cf.br ^bb151(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb151(%332: memref<32x64xi8>):  // 3 preds: ^bb148, ^bb149, ^bb150
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %333 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %334 = arith.index_cast %333 : i32 to index
      %335 = arith.index_cast %334 : index to i32
      cf.switch %335 : i32, [
        default: ^bb154,
        0: ^bb152,
        1: ^bb153
      ]
    ^bb152:  // pred: ^bb151
      cf.br ^bb155(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb153:  // pred: ^bb151
      cf.br ^bb155(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb154:  // pred: ^bb151
      cf.br ^bb155(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb155(%336: memref<64x128xi8>):  // 3 preds: ^bb152, ^bb153, ^bb154
      func.call @matmul_scalar_i8_i16(%332, %336, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %337 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %338 = arith.addi %337, %c1_i32 : i32
      %339 = arith.cmpi sge, %338, %c2_i32 : i32
      %340 = arith.subi %338, %c2_i32 : i32
      %341 = arith.select %339, %340, %338 : i32
      memref.store %341, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %342 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %343 = arith.addi %342, %c1_i32 : i32
      %344 = arith.cmpi sge, %343, %c2_i32 : i32
      %345 = arith.subi %343, %c2_i32 : i32
      %346 = arith.select %344, %345, %343 : i32
      memref.store %346, %_anonymous0[%c2] : memref<3xi32>
      %347 = arith.addi %327, %c1 : index
      cf.br ^bb146(%347 : index)
    ^bb156:  // pred: ^bb146
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %348 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %349 = arith.addi %348, %c1_i32 : i32
      %350 = arith.cmpi sge, %349, %c1_i32 : i32
      %351 = arith.select %350, %348, %349 : i32
      memref.store %351, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb157(%c0 : index)
    ^bb157(%352: index):  // 2 preds: ^bb156, ^bb166
      %353 = arith.cmpi slt, %352, %c16 : index
      cf.cond_br %353, ^bb158, ^bb167
    ^bb158:  // pred: ^bb157
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %354 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %355 = arith.index_cast %354 : i32 to index
      %356 = arith.index_cast %355 : index to i32
      cf.switch %356 : i32, [
        default: ^bb161,
        0: ^bb159,
        1: ^bb160
      ]
    ^bb159:  // pred: ^bb158
      cf.br ^bb162(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb160:  // pred: ^bb158
      cf.br ^bb162(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb161:  // pred: ^bb158
      cf.br ^bb162(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb162(%357: memref<32x64xi8>):  // 3 preds: ^bb159, ^bb160, ^bb161
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %358 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %359 = arith.index_cast %358 : i32 to index
      %360 = arith.index_cast %359 : index to i32
      cf.switch %360 : i32, [
        default: ^bb165,
        0: ^bb163,
        1: ^bb164
      ]
    ^bb163:  // pred: ^bb162
      cf.br ^bb166(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb164:  // pred: ^bb162
      cf.br ^bb166(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb165:  // pred: ^bb162
      cf.br ^bb166(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb166(%361: memref<64x128xi8>):  // 3 preds: ^bb163, ^bb164, ^bb165
      func.call @matmul_scalar_i8_i16(%357, %361, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %362 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %363 = arith.addi %362, %c1_i32 : i32
      %364 = arith.cmpi sge, %363, %c2_i32 : i32
      %365 = arith.subi %363, %c2_i32 : i32
      %366 = arith.select %364, %365, %363 : i32
      memref.store %366, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %367 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %368 = arith.addi %367, %c1_i32 : i32
      %369 = arith.cmpi sge, %368, %c2_i32 : i32
      %370 = arith.subi %368, %c2_i32 : i32
      %371 = arith.select %369, %370, %368 : i32
      memref.store %371, %_anonymous0[%c2] : memref<3xi32>
      %372 = arith.addi %352, %c1 : index
      cf.br ^bb157(%372 : index)
    ^bb167:  // pred: ^bb157
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %373 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %374 = arith.addi %373, %c1_i32 : i32
      %375 = arith.cmpi sge, %374, %c1_i32 : i32
      %376 = arith.select %375, %373, %374 : i32
      memref.store %376, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb168(%c0 : index)
    ^bb168(%377: index):  // 2 preds: ^bb167, ^bb177
      %378 = arith.cmpi slt, %377, %c16 : index
      cf.cond_br %378, ^bb169, ^bb178
    ^bb169:  // pred: ^bb168
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %379 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %380 = arith.index_cast %379 : i32 to index
      %381 = arith.index_cast %380 : index to i32
      cf.switch %381 : i32, [
        default: ^bb172,
        0: ^bb170,
        1: ^bb171
      ]
    ^bb170:  // pred: ^bb169
      cf.br ^bb173(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb171:  // pred: ^bb169
      cf.br ^bb173(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb172:  // pred: ^bb169
      cf.br ^bb173(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb173(%382: memref<32x64xi8>):  // 3 preds: ^bb170, ^bb171, ^bb172
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %383 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %384 = arith.index_cast %383 : i32 to index
      %385 = arith.index_cast %384 : index to i32
      cf.switch %385 : i32, [
        default: ^bb176,
        0: ^bb174,
        1: ^bb175
      ]
    ^bb174:  // pred: ^bb173
      cf.br ^bb177(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb175:  // pred: ^bb173
      cf.br ^bb177(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb176:  // pred: ^bb173
      cf.br ^bb177(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb177(%386: memref<64x128xi8>):  // 3 preds: ^bb174, ^bb175, ^bb176
      func.call @matmul_scalar_i8_i16(%382, %386, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %387 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %388 = arith.addi %387, %c1_i32 : i32
      %389 = arith.cmpi sge, %388, %c2_i32 : i32
      %390 = arith.subi %388, %c2_i32 : i32
      %391 = arith.select %389, %390, %388 : i32
      memref.store %391, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %392 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %393 = arith.addi %392, %c1_i32 : i32
      %394 = arith.cmpi sge, %393, %c2_i32 : i32
      %395 = arith.subi %393, %c2_i32 : i32
      %396 = arith.select %394, %395, %393 : i32
      memref.store %396, %_anonymous0[%c2] : memref<3xi32>
      %397 = arith.addi %377, %c1 : index
      cf.br ^bb168(%397 : index)
    ^bb178:  // pred: ^bb168
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %398 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %399 = arith.addi %398, %c1_i32 : i32
      %400 = arith.cmpi sge, %399, %c1_i32 : i32
      %401 = arith.select %400, %398, %399 : i32
      memref.store %401, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb179(%c0 : index)
    ^bb179(%402: index):  // 2 preds: ^bb178, ^bb188
      %403 = arith.cmpi slt, %402, %c16 : index
      cf.cond_br %403, ^bb180, ^bb189
    ^bb180:  // pred: ^bb179
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %404 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %405 = arith.index_cast %404 : i32 to index
      %406 = arith.index_cast %405 : index to i32
      cf.switch %406 : i32, [
        default: ^bb183,
        0: ^bb181,
        1: ^bb182
      ]
    ^bb181:  // pred: ^bb180
      cf.br ^bb184(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb182:  // pred: ^bb180
      cf.br ^bb184(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb183:  // pred: ^bb180
      cf.br ^bb184(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb184(%407: memref<32x64xi8>):  // 3 preds: ^bb181, ^bb182, ^bb183
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %408 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %409 = arith.index_cast %408 : i32 to index
      %410 = arith.index_cast %409 : index to i32
      cf.switch %410 : i32, [
        default: ^bb187,
        0: ^bb185,
        1: ^bb186
      ]
    ^bb185:  // pred: ^bb184
      cf.br ^bb188(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb186:  // pred: ^bb184
      cf.br ^bb188(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb187:  // pred: ^bb184
      cf.br ^bb188(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb188(%411: memref<64x128xi8>):  // 3 preds: ^bb185, ^bb186, ^bb187
      func.call @matmul_scalar_i8_i16(%407, %411, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %412 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %413 = arith.addi %412, %c1_i32 : i32
      %414 = arith.cmpi sge, %413, %c2_i32 : i32
      %415 = arith.subi %413, %c2_i32 : i32
      %416 = arith.select %414, %415, %413 : i32
      memref.store %416, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %417 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %418 = arith.addi %417, %c1_i32 : i32
      %419 = arith.cmpi sge, %418, %c2_i32 : i32
      %420 = arith.subi %418, %c2_i32 : i32
      %421 = arith.select %419, %420, %418 : i32
      memref.store %421, %_anonymous0[%c2] : memref<3xi32>
      %422 = arith.addi %402, %c1 : index
      cf.br ^bb179(%422 : index)
    ^bb189:  // pred: ^bb179
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %423 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %424 = arith.addi %423, %c1_i32 : i32
      %425 = arith.cmpi sge, %424, %c1_i32 : i32
      %426 = arith.select %425, %423, %424 : i32
      memref.store %426, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb190(%c0 : index)
    ^bb190(%427: index):  // 2 preds: ^bb189, ^bb199
      %428 = arith.cmpi slt, %427, %c16 : index
      cf.cond_br %428, ^bb191, ^bb200
    ^bb191:  // pred: ^bb190
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %429 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %430 = arith.index_cast %429 : i32 to index
      %431 = arith.index_cast %430 : index to i32
      cf.switch %431 : i32, [
        default: ^bb194,
        0: ^bb192,
        1: ^bb193
      ]
    ^bb192:  // pred: ^bb191
      cf.br ^bb195(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb193:  // pred: ^bb191
      cf.br ^bb195(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb194:  // pred: ^bb191
      cf.br ^bb195(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb195(%432: memref<32x64xi8>):  // 3 preds: ^bb192, ^bb193, ^bb194
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %433 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %434 = arith.index_cast %433 : i32 to index
      %435 = arith.index_cast %434 : index to i32
      cf.switch %435 : i32, [
        default: ^bb198,
        0: ^bb196,
        1: ^bb197
      ]
    ^bb196:  // pred: ^bb195
      cf.br ^bb199(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb197:  // pred: ^bb195
      cf.br ^bb199(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb198:  // pred: ^bb195
      cf.br ^bb199(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb199(%436: memref<64x128xi8>):  // 3 preds: ^bb196, ^bb197, ^bb198
      func.call @matmul_scalar_i8_i16(%432, %436, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %437 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %438 = arith.addi %437, %c1_i32 : i32
      %439 = arith.cmpi sge, %438, %c2_i32 : i32
      %440 = arith.subi %438, %c2_i32 : i32
      %441 = arith.select %439, %440, %438 : i32
      memref.store %441, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %442 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %443 = arith.addi %442, %c1_i32 : i32
      %444 = arith.cmpi sge, %443, %c2_i32 : i32
      %445 = arith.subi %443, %c2_i32 : i32
      %446 = arith.select %444, %445, %443 : i32
      memref.store %446, %_anonymous0[%c2] : memref<3xi32>
      %447 = arith.addi %427, %c1 : index
      cf.br ^bb190(%447 : index)
    ^bb200:  // pred: ^bb190
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %448 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %449 = arith.addi %448, %c1_i32 : i32
      %450 = arith.cmpi sge, %449, %c1_i32 : i32
      %451 = arith.select %450, %448, %449 : i32
      memref.store %451, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb201(%c0 : index)
    ^bb201(%452: index):  // 2 preds: ^bb200, ^bb210
      %453 = arith.cmpi slt, %452, %c16 : index
      cf.cond_br %453, ^bb202, ^bb211
    ^bb202:  // pred: ^bb201
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %454 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %455 = arith.index_cast %454 : i32 to index
      %456 = arith.index_cast %455 : index to i32
      cf.switch %456 : i32, [
        default: ^bb205,
        0: ^bb203,
        1: ^bb204
      ]
    ^bb203:  // pred: ^bb202
      cf.br ^bb206(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb204:  // pred: ^bb202
      cf.br ^bb206(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb205:  // pred: ^bb202
      cf.br ^bb206(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb206(%457: memref<32x64xi8>):  // 3 preds: ^bb203, ^bb204, ^bb205
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %458 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %459 = arith.index_cast %458 : i32 to index
      %460 = arith.index_cast %459 : index to i32
      cf.switch %460 : i32, [
        default: ^bb209,
        0: ^bb207,
        1: ^bb208
      ]
    ^bb207:  // pred: ^bb206
      cf.br ^bb210(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb208:  // pred: ^bb206
      cf.br ^bb210(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb209:  // pred: ^bb206
      cf.br ^bb210(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb210(%461: memref<64x128xi8>):  // 3 preds: ^bb207, ^bb208, ^bb209
      func.call @matmul_scalar_i8_i16(%457, %461, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %462 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %463 = arith.addi %462, %c1_i32 : i32
      %464 = arith.cmpi sge, %463, %c2_i32 : i32
      %465 = arith.subi %463, %c2_i32 : i32
      %466 = arith.select %464, %465, %463 : i32
      memref.store %466, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %467 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %468 = arith.addi %467, %c1_i32 : i32
      %469 = arith.cmpi sge, %468, %c2_i32 : i32
      %470 = arith.subi %468, %c2_i32 : i32
      %471 = arith.select %469, %470, %468 : i32
      memref.store %471, %_anonymous0[%c2] : memref<3xi32>
      %472 = arith.addi %452, %c1 : index
      cf.br ^bb201(%472 : index)
    ^bb211:  // pred: ^bb201
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %473 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %474 = arith.addi %473, %c1_i32 : i32
      %475 = arith.cmpi sge, %474, %c1_i32 : i32
      %476 = arith.select %475, %473, %474 : i32
      memref.store %476, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb212(%c0 : index)
    ^bb212(%477: index):  // 2 preds: ^bb211, ^bb221
      %478 = arith.cmpi slt, %477, %c16 : index
      cf.cond_br %478, ^bb213, ^bb222
    ^bb213:  // pred: ^bb212
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %479 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %480 = arith.index_cast %479 : i32 to index
      %481 = arith.index_cast %480 : index to i32
      cf.switch %481 : i32, [
        default: ^bb216,
        0: ^bb214,
        1: ^bb215
      ]
    ^bb214:  // pred: ^bb213
      cf.br ^bb217(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb215:  // pred: ^bb213
      cf.br ^bb217(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb216:  // pred: ^bb213
      cf.br ^bb217(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb217(%482: memref<32x64xi8>):  // 3 preds: ^bb214, ^bb215, ^bb216
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %483 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %484 = arith.index_cast %483 : i32 to index
      %485 = arith.index_cast %484 : index to i32
      cf.switch %485 : i32, [
        default: ^bb220,
        0: ^bb218,
        1: ^bb219
      ]
    ^bb218:  // pred: ^bb217
      cf.br ^bb221(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb219:  // pred: ^bb217
      cf.br ^bb221(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb220:  // pred: ^bb217
      cf.br ^bb221(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb221(%486: memref<64x128xi8>):  // 3 preds: ^bb218, ^bb219, ^bb220
      func.call @matmul_scalar_i8_i16(%482, %486, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %487 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %488 = arith.addi %487, %c1_i32 : i32
      %489 = arith.cmpi sge, %488, %c2_i32 : i32
      %490 = arith.subi %488, %c2_i32 : i32
      %491 = arith.select %489, %490, %488 : i32
      memref.store %491, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %492 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %493 = arith.addi %492, %c1_i32 : i32
      %494 = arith.cmpi sge, %493, %c2_i32 : i32
      %495 = arith.subi %493, %c2_i32 : i32
      %496 = arith.select %494, %495, %493 : i32
      memref.store %496, %_anonymous0[%c2] : memref<3xi32>
      %497 = arith.addi %477, %c1 : index
      cf.br ^bb212(%497 : index)
    ^bb222:  // pred: ^bb212
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %498 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %499 = arith.addi %498, %c1_i32 : i32
      %500 = arith.cmpi sge, %499, %c1_i32 : i32
      %501 = arith.select %500, %498, %499 : i32
      memref.store %501, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb223(%c0 : index)
    ^bb223(%502: index):  // 2 preds: ^bb222, ^bb232
      %503 = arith.cmpi slt, %502, %c16 : index
      cf.cond_br %503, ^bb224, ^bb233
    ^bb224:  // pred: ^bb223
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %504 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %505 = arith.index_cast %504 : i32 to index
      %506 = arith.index_cast %505 : index to i32
      cf.switch %506 : i32, [
        default: ^bb227,
        0: ^bb225,
        1: ^bb226
      ]
    ^bb225:  // pred: ^bb224
      cf.br ^bb228(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb226:  // pred: ^bb224
      cf.br ^bb228(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb227:  // pred: ^bb224
      cf.br ^bb228(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb228(%507: memref<32x64xi8>):  // 3 preds: ^bb225, ^bb226, ^bb227
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %508 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %509 = arith.index_cast %508 : i32 to index
      %510 = arith.index_cast %509 : index to i32
      cf.switch %510 : i32, [
        default: ^bb231,
        0: ^bb229,
        1: ^bb230
      ]
    ^bb229:  // pred: ^bb228
      cf.br ^bb232(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb230:  // pred: ^bb228
      cf.br ^bb232(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb231:  // pred: ^bb228
      cf.br ^bb232(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb232(%511: memref<64x128xi8>):  // 3 preds: ^bb229, ^bb230, ^bb231
      func.call @matmul_scalar_i8_i16(%507, %511, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %512 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %513 = arith.addi %512, %c1_i32 : i32
      %514 = arith.cmpi sge, %513, %c2_i32 : i32
      %515 = arith.subi %513, %c2_i32 : i32
      %516 = arith.select %514, %515, %513 : i32
      memref.store %516, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %517 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %518 = arith.addi %517, %c1_i32 : i32
      %519 = arith.cmpi sge, %518, %c2_i32 : i32
      %520 = arith.subi %518, %c2_i32 : i32
      %521 = arith.select %519, %520, %518 : i32
      memref.store %521, %_anonymous0[%c2] : memref<3xi32>
      %522 = arith.addi %502, %c1 : index
      cf.br ^bb223(%522 : index)
    ^bb233:  // pred: ^bb223
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %523 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %524 = arith.addi %523, %c1_i32 : i32
      %525 = arith.cmpi sge, %524, %c1_i32 : i32
      %526 = arith.select %525, %523, %524 : i32
      memref.store %526, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb234(%c0 : index)
    ^bb234(%527: index):  // 2 preds: ^bb233, ^bb243
      %528 = arith.cmpi slt, %527, %c16 : index
      cf.cond_br %528, ^bb235, ^bb244
    ^bb235:  // pred: ^bb234
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %529 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %530 = arith.index_cast %529 : i32 to index
      %531 = arith.index_cast %530 : index to i32
      cf.switch %531 : i32, [
        default: ^bb238,
        0: ^bb236,
        1: ^bb237
      ]
    ^bb236:  // pred: ^bb235
      cf.br ^bb239(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb237:  // pred: ^bb235
      cf.br ^bb239(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb238:  // pred: ^bb235
      cf.br ^bb239(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb239(%532: memref<32x64xi8>):  // 3 preds: ^bb236, ^bb237, ^bb238
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %533 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %534 = arith.index_cast %533 : i32 to index
      %535 = arith.index_cast %534 : index to i32
      cf.switch %535 : i32, [
        default: ^bb242,
        0: ^bb240,
        1: ^bb241
      ]
    ^bb240:  // pred: ^bb239
      cf.br ^bb243(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb241:  // pred: ^bb239
      cf.br ^bb243(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb242:  // pred: ^bb239
      cf.br ^bb243(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb243(%536: memref<64x128xi8>):  // 3 preds: ^bb240, ^bb241, ^bb242
      func.call @matmul_scalar_i8_i16(%532, %536, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %537 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %538 = arith.addi %537, %c1_i32 : i32
      %539 = arith.cmpi sge, %538, %c2_i32 : i32
      %540 = arith.subi %538, %c2_i32 : i32
      %541 = arith.select %539, %540, %538 : i32
      memref.store %541, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %542 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %543 = arith.addi %542, %c1_i32 : i32
      %544 = arith.cmpi sge, %543, %c2_i32 : i32
      %545 = arith.subi %543, %c2_i32 : i32
      %546 = arith.select %544, %545, %543 : i32
      memref.store %546, %_anonymous0[%c2] : memref<3xi32>
      %547 = arith.addi %527, %c1 : index
      cf.br ^bb234(%547 : index)
    ^bb244:  // pred: ^bb234
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %548 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %549 = arith.addi %548, %c1_i32 : i32
      %550 = arith.cmpi sge, %549, %c1_i32 : i32
      %551 = arith.select %550, %548, %549 : i32
      memref.store %551, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb245(%c0 : index)
    ^bb245(%552: index):  // 2 preds: ^bb244, ^bb254
      %553 = arith.cmpi slt, %552, %c16 : index
      cf.cond_br %553, ^bb246, ^bb255
    ^bb246:  // pred: ^bb245
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %554 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %555 = arith.index_cast %554 : i32 to index
      %556 = arith.index_cast %555 : index to i32
      cf.switch %556 : i32, [
        default: ^bb249,
        0: ^bb247,
        1: ^bb248
      ]
    ^bb247:  // pred: ^bb246
      cf.br ^bb250(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb248:  // pred: ^bb246
      cf.br ^bb250(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb249:  // pred: ^bb246
      cf.br ^bb250(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb250(%557: memref<32x64xi8>):  // 3 preds: ^bb247, ^bb248, ^bb249
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %558 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %559 = arith.index_cast %558 : i32 to index
      %560 = arith.index_cast %559 : index to i32
      cf.switch %560 : i32, [
        default: ^bb253,
        0: ^bb251,
        1: ^bb252
      ]
    ^bb251:  // pred: ^bb250
      cf.br ^bb254(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb252:  // pred: ^bb250
      cf.br ^bb254(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb253:  // pred: ^bb250
      cf.br ^bb254(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb254(%561: memref<64x128xi8>):  // 3 preds: ^bb251, ^bb252, ^bb253
      func.call @matmul_scalar_i8_i16(%557, %561, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %562 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %563 = arith.addi %562, %c1_i32 : i32
      %564 = arith.cmpi sge, %563, %c2_i32 : i32
      %565 = arith.subi %563, %c2_i32 : i32
      %566 = arith.select %564, %565, %563 : i32
      memref.store %566, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %567 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %568 = arith.addi %567, %c1_i32 : i32
      %569 = arith.cmpi sge, %568, %c2_i32 : i32
      %570 = arith.subi %568, %c2_i32 : i32
      %571 = arith.select %569, %570, %568 : i32
      memref.store %571, %_anonymous0[%c2] : memref<3xi32>
      %572 = arith.addi %552, %c1 : index
      cf.br ^bb245(%572 : index)
    ^bb255:  // pred: ^bb245
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %573 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %574 = arith.addi %573, %c1_i32 : i32
      %575 = arith.cmpi sge, %574, %c1_i32 : i32
      %576 = arith.select %575, %573, %574 : i32
      memref.store %576, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb256(%c0 : index)
    ^bb256(%577: index):  // 2 preds: ^bb255, ^bb265
      %578 = arith.cmpi slt, %577, %c16 : index
      cf.cond_br %578, ^bb257, ^bb266
    ^bb257:  // pred: ^bb256
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %579 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %580 = arith.index_cast %579 : i32 to index
      %581 = arith.index_cast %580 : index to i32
      cf.switch %581 : i32, [
        default: ^bb260,
        0: ^bb258,
        1: ^bb259
      ]
    ^bb258:  // pred: ^bb257
      cf.br ^bb261(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb259:  // pred: ^bb257
      cf.br ^bb261(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb260:  // pred: ^bb257
      cf.br ^bb261(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb261(%582: memref<32x64xi8>):  // 3 preds: ^bb258, ^bb259, ^bb260
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %583 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %584 = arith.index_cast %583 : i32 to index
      %585 = arith.index_cast %584 : index to i32
      cf.switch %585 : i32, [
        default: ^bb264,
        0: ^bb262,
        1: ^bb263
      ]
    ^bb262:  // pred: ^bb261
      cf.br ^bb265(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb263:  // pred: ^bb261
      cf.br ^bb265(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb264:  // pred: ^bb261
      cf.br ^bb265(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb265(%586: memref<64x128xi8>):  // 3 preds: ^bb262, ^bb263, ^bb264
      func.call @matmul_scalar_i8_i16(%582, %586, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %587 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %588 = arith.addi %587, %c1_i32 : i32
      %589 = arith.cmpi sge, %588, %c2_i32 : i32
      %590 = arith.subi %588, %c2_i32 : i32
      %591 = arith.select %589, %590, %588 : i32
      memref.store %591, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %592 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %593 = arith.addi %592, %c1_i32 : i32
      %594 = arith.cmpi sge, %593, %c2_i32 : i32
      %595 = arith.subi %593, %c2_i32 : i32
      %596 = arith.select %594, %595, %593 : i32
      memref.store %596, %_anonymous0[%c2] : memref<3xi32>
      %597 = arith.addi %577, %c1 : index
      cf.br ^bb256(%597 : index)
    ^bb266:  // pred: ^bb256
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %598 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %599 = arith.addi %598, %c1_i32 : i32
      %600 = arith.cmpi sge, %599, %c1_i32 : i32
      %601 = arith.select %600, %598, %599 : i32
      memref.store %601, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb267(%c0 : index)
    ^bb267(%602: index):  // 2 preds: ^bb266, ^bb276
      %603 = arith.cmpi slt, %602, %c16 : index
      cf.cond_br %603, ^bb268, ^bb277
    ^bb268:  // pred: ^bb267
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %604 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %605 = arith.index_cast %604 : i32 to index
      %606 = arith.index_cast %605 : index to i32
      cf.switch %606 : i32, [
        default: ^bb271,
        0: ^bb269,
        1: ^bb270
      ]
    ^bb269:  // pred: ^bb268
      cf.br ^bb272(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb270:  // pred: ^bb268
      cf.br ^bb272(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb271:  // pred: ^bb268
      cf.br ^bb272(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb272(%607: memref<32x64xi8>):  // 3 preds: ^bb269, ^bb270, ^bb271
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %608 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %609 = arith.index_cast %608 : i32 to index
      %610 = arith.index_cast %609 : index to i32
      cf.switch %610 : i32, [
        default: ^bb275,
        0: ^bb273,
        1: ^bb274
      ]
    ^bb273:  // pred: ^bb272
      cf.br ^bb276(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb274:  // pred: ^bb272
      cf.br ^bb276(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb275:  // pred: ^bb272
      cf.br ^bb276(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb276(%611: memref<64x128xi8>):  // 3 preds: ^bb273, ^bb274, ^bb275
      func.call @matmul_scalar_i8_i16(%607, %611, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %612 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %613 = arith.addi %612, %c1_i32 : i32
      %614 = arith.cmpi sge, %613, %c2_i32 : i32
      %615 = arith.subi %613, %c2_i32 : i32
      %616 = arith.select %614, %615, %613 : i32
      memref.store %616, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %617 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %618 = arith.addi %617, %c1_i32 : i32
      %619 = arith.cmpi sge, %618, %c2_i32 : i32
      %620 = arith.subi %618, %c2_i32 : i32
      %621 = arith.select %619, %620, %618 : i32
      memref.store %621, %_anonymous0[%c2] : memref<3xi32>
      %622 = arith.addi %602, %c1 : index
      cf.br ^bb267(%622 : index)
    ^bb277:  // pred: ^bb267
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %623 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %624 = arith.addi %623, %c1_i32 : i32
      %625 = arith.cmpi sge, %624, %c1_i32 : i32
      %626 = arith.select %625, %623, %624 : i32
      memref.store %626, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb278(%c0 : index)
    ^bb278(%627: index):  // 2 preds: ^bb277, ^bb287
      %628 = arith.cmpi slt, %627, %c16 : index
      cf.cond_br %628, ^bb279, ^bb288
    ^bb279:  // pred: ^bb278
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %629 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %630 = arith.index_cast %629 : i32 to index
      %631 = arith.index_cast %630 : index to i32
      cf.switch %631 : i32, [
        default: ^bb282,
        0: ^bb280,
        1: ^bb281
      ]
    ^bb280:  // pred: ^bb279
      cf.br ^bb283(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb281:  // pred: ^bb279
      cf.br ^bb283(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb282:  // pred: ^bb279
      cf.br ^bb283(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb283(%632: memref<32x64xi8>):  // 3 preds: ^bb280, ^bb281, ^bb282
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %633 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %634 = arith.index_cast %633 : i32 to index
      %635 = arith.index_cast %634 : index to i32
      cf.switch %635 : i32, [
        default: ^bb286,
        0: ^bb284,
        1: ^bb285
      ]
    ^bb284:  // pred: ^bb283
      cf.br ^bb287(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb285:  // pred: ^bb283
      cf.br ^bb287(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb286:  // pred: ^bb283
      cf.br ^bb287(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb287(%636: memref<64x128xi8>):  // 3 preds: ^bb284, ^bb285, ^bb286
      func.call @matmul_scalar_i8_i16(%632, %636, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %637 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %638 = arith.addi %637, %c1_i32 : i32
      %639 = arith.cmpi sge, %638, %c2_i32 : i32
      %640 = arith.subi %638, %c2_i32 : i32
      %641 = arith.select %639, %640, %638 : i32
      memref.store %641, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %642 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %643 = arith.addi %642, %c1_i32 : i32
      %644 = arith.cmpi sge, %643, %c2_i32 : i32
      %645 = arith.subi %643, %c2_i32 : i32
      %646 = arith.select %644, %645, %643 : i32
      memref.store %646, %_anonymous0[%c2] : memref<3xi32>
      %647 = arith.addi %627, %c1 : index
      cf.br ^bb278(%647 : index)
    ^bb288:  // pred: ^bb278
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %648 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %649 = arith.addi %648, %c1_i32 : i32
      %650 = arith.cmpi sge, %649, %c1_i32 : i32
      %651 = arith.select %650, %648, %649 : i32
      memref.store %651, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb289(%c0 : index)
    ^bb289(%652: index):  // 2 preds: ^bb288, ^bb298
      %653 = arith.cmpi slt, %652, %c16 : index
      cf.cond_br %653, ^bb290, ^bb299
    ^bb290:  // pred: ^bb289
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %654 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %655 = arith.index_cast %654 : i32 to index
      %656 = arith.index_cast %655 : index to i32
      cf.switch %656 : i32, [
        default: ^bb293,
        0: ^bb291,
        1: ^bb292
      ]
    ^bb291:  // pred: ^bb290
      cf.br ^bb294(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb292:  // pred: ^bb290
      cf.br ^bb294(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb293:  // pred: ^bb290
      cf.br ^bb294(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb294(%657: memref<32x64xi8>):  // 3 preds: ^bb291, ^bb292, ^bb293
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %658 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %659 = arith.index_cast %658 : i32 to index
      %660 = arith.index_cast %659 : index to i32
      cf.switch %660 : i32, [
        default: ^bb297,
        0: ^bb295,
        1: ^bb296
      ]
    ^bb295:  // pred: ^bb294
      cf.br ^bb298(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb296:  // pred: ^bb294
      cf.br ^bb298(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb297:  // pred: ^bb294
      cf.br ^bb298(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb298(%661: memref<64x128xi8>):  // 3 preds: ^bb295, ^bb296, ^bb297
      func.call @matmul_scalar_i8_i16(%657, %661, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %662 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %663 = arith.addi %662, %c1_i32 : i32
      %664 = arith.cmpi sge, %663, %c2_i32 : i32
      %665 = arith.subi %663, %c2_i32 : i32
      %666 = arith.select %664, %665, %663 : i32
      memref.store %666, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %667 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %668 = arith.addi %667, %c1_i32 : i32
      %669 = arith.cmpi sge, %668, %c2_i32 : i32
      %670 = arith.subi %668, %c2_i32 : i32
      %671 = arith.select %669, %670, %668 : i32
      memref.store %671, %_anonymous0[%c2] : memref<3xi32>
      %672 = arith.addi %652, %c1 : index
      cf.br ^bb289(%672 : index)
    ^bb299:  // pred: ^bb289
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %673 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %674 = arith.addi %673, %c1_i32 : i32
      %675 = arith.cmpi sge, %674, %c1_i32 : i32
      %676 = arith.select %675, %673, %674 : i32
      memref.store %676, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb300(%c0 : index)
    ^bb300(%677: index):  // 2 preds: ^bb299, ^bb309
      %678 = arith.cmpi slt, %677, %c16 : index
      cf.cond_br %678, ^bb301, ^bb310
    ^bb301:  // pred: ^bb300
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %679 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %680 = arith.index_cast %679 : i32 to index
      %681 = arith.index_cast %680 : index to i32
      cf.switch %681 : i32, [
        default: ^bb304,
        0: ^bb302,
        1: ^bb303
      ]
    ^bb302:  // pred: ^bb301
      cf.br ^bb305(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb303:  // pred: ^bb301
      cf.br ^bb305(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb304:  // pred: ^bb301
      cf.br ^bb305(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb305(%682: memref<32x64xi8>):  // 3 preds: ^bb302, ^bb303, ^bb304
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %683 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %684 = arith.index_cast %683 : i32 to index
      %685 = arith.index_cast %684 : index to i32
      cf.switch %685 : i32, [
        default: ^bb308,
        0: ^bb306,
        1: ^bb307
      ]
    ^bb306:  // pred: ^bb305
      cf.br ^bb309(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb307:  // pred: ^bb305
      cf.br ^bb309(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb308:  // pred: ^bb305
      cf.br ^bb309(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb309(%686: memref<64x128xi8>):  // 3 preds: ^bb306, ^bb307, ^bb308
      func.call @matmul_scalar_i8_i16(%682, %686, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %687 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %688 = arith.addi %687, %c1_i32 : i32
      %689 = arith.cmpi sge, %688, %c2_i32 : i32
      %690 = arith.subi %688, %c2_i32 : i32
      %691 = arith.select %689, %690, %688 : i32
      memref.store %691, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %692 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %693 = arith.addi %692, %c1_i32 : i32
      %694 = arith.cmpi sge, %693, %c2_i32 : i32
      %695 = arith.subi %693, %c2_i32 : i32
      %696 = arith.select %694, %695, %693 : i32
      memref.store %696, %_anonymous0[%c2] : memref<3xi32>
      %697 = arith.addi %677, %c1 : index
      cf.br ^bb300(%697 : index)
    ^bb310:  // pred: ^bb300
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %698 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %699 = arith.addi %698, %c1_i32 : i32
      %700 = arith.cmpi sge, %699, %c1_i32 : i32
      %701 = arith.select %700, %698, %699 : i32
      memref.store %701, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb311(%c0 : index)
    ^bb311(%702: index):  // 2 preds: ^bb310, ^bb320
      %703 = arith.cmpi slt, %702, %c16 : index
      cf.cond_br %703, ^bb312, ^bb321
    ^bb312:  // pred: ^bb311
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %704 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %705 = arith.index_cast %704 : i32 to index
      %706 = arith.index_cast %705 : index to i32
      cf.switch %706 : i32, [
        default: ^bb315,
        0: ^bb313,
        1: ^bb314
      ]
    ^bb313:  // pred: ^bb312
      cf.br ^bb316(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb314:  // pred: ^bb312
      cf.br ^bb316(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb315:  // pred: ^bb312
      cf.br ^bb316(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb316(%707: memref<32x64xi8>):  // 3 preds: ^bb313, ^bb314, ^bb315
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %708 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %709 = arith.index_cast %708 : i32 to index
      %710 = arith.index_cast %709 : index to i32
      cf.switch %710 : i32, [
        default: ^bb319,
        0: ^bb317,
        1: ^bb318
      ]
    ^bb317:  // pred: ^bb316
      cf.br ^bb320(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb318:  // pred: ^bb316
      cf.br ^bb320(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb319:  // pred: ^bb316
      cf.br ^bb320(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb320(%711: memref<64x128xi8>):  // 3 preds: ^bb317, ^bb318, ^bb319
      func.call @matmul_scalar_i8_i16(%707, %711, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %712 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %713 = arith.addi %712, %c1_i32 : i32
      %714 = arith.cmpi sge, %713, %c2_i32 : i32
      %715 = arith.subi %713, %c2_i32 : i32
      %716 = arith.select %714, %715, %713 : i32
      memref.store %716, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %717 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %718 = arith.addi %717, %c1_i32 : i32
      %719 = arith.cmpi sge, %718, %c2_i32 : i32
      %720 = arith.subi %718, %c2_i32 : i32
      %721 = arith.select %719, %720, %718 : i32
      memref.store %721, %_anonymous0[%c2] : memref<3xi32>
      %722 = arith.addi %702, %c1 : index
      cf.br ^bb311(%722 : index)
    ^bb321:  // pred: ^bb311
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %723 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %724 = arith.addi %723, %c1_i32 : i32
      %725 = arith.cmpi sge, %724, %c1_i32 : i32
      %726 = arith.select %725, %723, %724 : i32
      memref.store %726, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb322(%c0 : index)
    ^bb322(%727: index):  // 2 preds: ^bb321, ^bb331
      %728 = arith.cmpi slt, %727, %c16 : index
      cf.cond_br %728, ^bb323, ^bb332
    ^bb323:  // pred: ^bb322
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %729 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %730 = arith.index_cast %729 : i32 to index
      %731 = arith.index_cast %730 : index to i32
      cf.switch %731 : i32, [
        default: ^bb326,
        0: ^bb324,
        1: ^bb325
      ]
    ^bb324:  // pred: ^bb323
      cf.br ^bb327(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb325:  // pred: ^bb323
      cf.br ^bb327(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb326:  // pred: ^bb323
      cf.br ^bb327(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb327(%732: memref<32x64xi8>):  // 3 preds: ^bb324, ^bb325, ^bb326
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %733 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %734 = arith.index_cast %733 : i32 to index
      %735 = arith.index_cast %734 : index to i32
      cf.switch %735 : i32, [
        default: ^bb330,
        0: ^bb328,
        1: ^bb329
      ]
    ^bb328:  // pred: ^bb327
      cf.br ^bb331(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb329:  // pred: ^bb327
      cf.br ^bb331(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb330:  // pred: ^bb327
      cf.br ^bb331(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb331(%736: memref<64x128xi8>):  // 3 preds: ^bb328, ^bb329, ^bb330
      func.call @matmul_scalar_i8_i16(%732, %736, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %737 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %738 = arith.addi %737, %c1_i32 : i32
      %739 = arith.cmpi sge, %738, %c2_i32 : i32
      %740 = arith.subi %738, %c2_i32 : i32
      %741 = arith.select %739, %740, %738 : i32
      memref.store %741, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %742 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %743 = arith.addi %742, %c1_i32 : i32
      %744 = arith.cmpi sge, %743, %c2_i32 : i32
      %745 = arith.subi %743, %c2_i32 : i32
      %746 = arith.select %744, %745, %743 : i32
      memref.store %746, %_anonymous0[%c2] : memref<3xi32>
      %747 = arith.addi %727, %c1 : index
      cf.br ^bb322(%747 : index)
    ^bb332:  // pred: ^bb322
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %748 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %749 = arith.addi %748, %c1_i32 : i32
      %750 = arith.cmpi sge, %749, %c1_i32 : i32
      %751 = arith.select %750, %748, %749 : i32
      memref.store %751, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb333(%c0 : index)
    ^bb333(%752: index):  // 2 preds: ^bb332, ^bb342
      %753 = arith.cmpi slt, %752, %c16 : index
      cf.cond_br %753, ^bb334, ^bb343
    ^bb334:  // pred: ^bb333
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %754 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %755 = arith.index_cast %754 : i32 to index
      %756 = arith.index_cast %755 : index to i32
      cf.switch %756 : i32, [
        default: ^bb337,
        0: ^bb335,
        1: ^bb336
      ]
    ^bb335:  // pred: ^bb334
      cf.br ^bb338(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb336:  // pred: ^bb334
      cf.br ^bb338(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb337:  // pred: ^bb334
      cf.br ^bb338(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb338(%757: memref<32x64xi8>):  // 3 preds: ^bb335, ^bb336, ^bb337
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %758 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %759 = arith.index_cast %758 : i32 to index
      %760 = arith.index_cast %759 : index to i32
      cf.switch %760 : i32, [
        default: ^bb341,
        0: ^bb339,
        1: ^bb340
      ]
    ^bb339:  // pred: ^bb338
      cf.br ^bb342(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb340:  // pred: ^bb338
      cf.br ^bb342(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb341:  // pred: ^bb338
      cf.br ^bb342(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb342(%761: memref<64x128xi8>):  // 3 preds: ^bb339, ^bb340, ^bb341
      func.call @matmul_scalar_i8_i16(%757, %761, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %762 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %763 = arith.addi %762, %c1_i32 : i32
      %764 = arith.cmpi sge, %763, %c2_i32 : i32
      %765 = arith.subi %763, %c2_i32 : i32
      %766 = arith.select %764, %765, %763 : i32
      memref.store %766, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %767 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %768 = arith.addi %767, %c1_i32 : i32
      %769 = arith.cmpi sge, %768, %c2_i32 : i32
      %770 = arith.subi %768, %c2_i32 : i32
      %771 = arith.select %769, %770, %768 : i32
      memref.store %771, %_anonymous0[%c2] : memref<3xi32>
      %772 = arith.addi %752, %c1 : index
      cf.br ^bb333(%772 : index)
    ^bb343:  // pred: ^bb333
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %773 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %774 = arith.addi %773, %c1_i32 : i32
      %775 = arith.cmpi sge, %774, %c1_i32 : i32
      %776 = arith.select %775, %773, %774 : i32
      memref.store %776, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb344(%c0 : index)
    ^bb344(%777: index):  // 2 preds: ^bb343, ^bb353
      %778 = arith.cmpi slt, %777, %c16 : index
      cf.cond_br %778, ^bb345, ^bb354
    ^bb345:  // pred: ^bb344
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %779 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %780 = arith.index_cast %779 : i32 to index
      %781 = arith.index_cast %780 : index to i32
      cf.switch %781 : i32, [
        default: ^bb348,
        0: ^bb346,
        1: ^bb347
      ]
    ^bb346:  // pred: ^bb345
      cf.br ^bb349(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb347:  // pred: ^bb345
      cf.br ^bb349(%A_L2L1_cons_buff_1 : memref<32x64xi8>)
    ^bb348:  // pred: ^bb345
      cf.br ^bb349(%A_L2L1_cons_buff_0 : memref<32x64xi8>)
    ^bb349(%782: memref<32x64xi8>):  // 3 preds: ^bb346, ^bb347, ^bb348
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %783 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %784 = arith.index_cast %783 : i32 to index
      %785 = arith.index_cast %784 : index to i32
      cf.switch %785 : i32, [
        default: ^bb352,
        0: ^bb350,
        1: ^bb351
      ]
    ^bb350:  // pred: ^bb349
      cf.br ^bb353(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb351:  // pred: ^bb349
      cf.br ^bb353(%B_L2L1_cons_buff_1 : memref<64x128xi8>)
    ^bb352:  // pred: ^bb349
      cf.br ^bb353(%B_L2L1_cons_buff_0 : memref<64x128xi8>)
    ^bb353(%786: memref<64x128xi8>):  // 3 preds: ^bb350, ^bb351, ^bb352
      func.call @matmul_scalar_i8_i16(%782, %786, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %787 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %788 = arith.addi %787, %c1_i32 : i32
      %789 = arith.cmpi sge, %788, %c2_i32 : i32
      %790 = arith.subi %788, %c2_i32 : i32
      %791 = arith.select %789, %790, %788 : i32
      memref.store %791, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %792 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %793 = arith.addi %792, %c1_i32 : i32
      %794 = arith.cmpi sge, %793, %c2_i32 : i32
      %795 = arith.subi %793, %c2_i32 : i32
      %796 = arith.select %794, %795, %793 : i32
      memref.store %796, %_anonymous0[%c2] : memref<3xi32>
      %797 = arith.addi %777, %c1 : index
      cf.br ^bb344(%797 : index)
    ^bb354:  // pred: ^bb344
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %798 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %799 = arith.addi %798, %c1_i32 : i32
      %800 = arith.cmpi sge, %799, %c1_i32 : i32
      %801 = arith.select %800, %798, %799 : i32
      memref.store %801, %_anonymous0[%c0] : memref<3xi32>
      %802 = arith.addi %0, %c1 : index
      cf.br ^bb1(%802 : index)
    ^bb355:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"]}
    aie.runtime_sequence @seq(%arg0: memref<131072xi8>, %arg1: memref<1048576xi8>, %arg2: memref<131072xi16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %3 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %6 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %9 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %12 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %15 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %18 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %21 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %24 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %27 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %30 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %33 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %36 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %39 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %42 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %45 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %48 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %51 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %54 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %57 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %60 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %63 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %66 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %69 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %72 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %75 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %78 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %81 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %84 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %87 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %90 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %93 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
    aie.shim_dma_allocation @A_L3L2_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<32x512xi8>, 0, 16384) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<32x512xi8>, 0, 16384) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<32x512xi8>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<32x512xi8>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%C_L1L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%C_L1L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @C_L2L3_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
