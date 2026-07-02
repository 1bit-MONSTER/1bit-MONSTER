module {
  aie.device(npu2) {
    func.func private @zero_scalar_i16(memref<128x128xi16>) attributes {link_with = "mm_128x64x128.o"}
    func.func private @matmul_scalar_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) attributes {link_with = "mm_128x64x128.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %C_L2L3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "C_L2L3_cons_prod_lock_0"}
    %C_L2L3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "C_L2L3_cons_cons_lock_0"}
    %C_L1L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, sym_name = "C_L1L2_cons_buff_0"} : memref<128x128xi16> 
    %C_L1L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 1 : i32, sym_name = "C_L1L2_cons_prod_lock_0"}
    %C_L1L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "C_L1L2_cons_cons_lock_0"}
    %C_L1L2_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "C_L1L2_buff_0"} : memref<128x128xi16> 
    %C_L1L2_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "C_L1L2_prod_lock_0"}
    %C_L1L2_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_L1L2_cons_lock_0"}
    %B_L2L1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 33792 : i32, sym_name = "B_L2L1_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 41984 : i32, sym_name = "B_L2L1_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "B_L2L1_cons_prod_lock_0"}
    %B_L2L1_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_L2L1_cons_cons_lock_0"}
    %B_L3L2_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "B_L3L2_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 8192 : i32, sym_name = "B_L3L2_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "B_L3L2_cons_prod_lock_0"}
    %B_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "B_L3L2_cons_cons_lock_0"}
    %B_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "B_L3L2_prod_lock_0"}
    %B_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "B_L3L2_cons_lock_0"}
    %A_L2L1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 50176 : i32, sym_name = "A_L2L1_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 52224 : i32, sym_name = "A_L2L1_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L2L1_cons_prod_lock_0"}
    %A_L2L1_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L2L1_cons_cons_lock_0"}
    %A_L3L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "A_L3L2_cons_buff_0"} : memref<128x512xi8> 
    %A_L3L2_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, sym_name = "A_L3L2_cons_buff_1"} : memref<128x512xi8> 
    %A_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "A_L3L2_cons_prod_lock_0"}
    %A_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "A_L3L2_cons_cons_lock_0"}
    %A_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L2_prod_lock_0"}
    %A_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L2_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %shim_noc_tile_0_0, DMA : 0)
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 54272 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb90
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb91
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<128x128xi16>) -> ()
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
      func.call @matmul_scalar_i8_i16(%7, %11, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<128x128xi16>) -> ()
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
      func.call @matmul_scalar_i8_i16(%32, %36, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<128x128xi16>) -> ()
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
      func.call @matmul_scalar_i8_i16(%57, %61, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<128x128xi16>) -> ()
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
      func.call @matmul_scalar_i8_i16(%82, %86, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<128x128xi16>) -> ()
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
      func.call @matmul_scalar_i8_i16(%107, %111, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<128x128xi16>) -> ()
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
      func.call @matmul_scalar_i8_i16(%132, %136, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<128x128xi16>) -> ()
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
      func.call @matmul_scalar_i8_i16(%157, %161, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      func.call @zero_scalar_i16(%C_L1L2_buff_0) : (memref<128x128xi16>) -> ()
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
      func.call @matmul_scalar_i8_i16(%182, %186, %C_L1L2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      %202 = arith.addi %0, %c1 : index
      cf.br ^bb1(%202 : index)
    ^bb91:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"]}
    aie.runtime_sequence @seq(%arg0: memref<131072xi8>, %arg1: memref<1048576xi8>, %arg2: memref<131072xi16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %3 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %6 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %9 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %12 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %15 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %18 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
      %21 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
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
    aie.shim_dma_allocation @A_L3L2_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<128x512xi8>, 0, 65536) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<128x512xi8>, 0, 65536) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<128x512xi8>, 0, 65536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<128x512xi8>, 0, 65536) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L1L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
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
      aie.dma_bd(%C_L1L2_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%B_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @C_L2L3_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
