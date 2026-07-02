; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-p:32:32:32:32:8-s0:512:512-a0:8:8-S512-n32:64-P1-p0:20:32:32:32:8-p1:20:32:32:32:8-p2:20:32:32:32:8-p3:20:32:32:32:8-p4:20:32:32:32:8-p5:20:32:32:32:8-p6:20:32:32:32:8-p7:20:32:32:32:8-p8:20:32:32:32:8-p9:20:32:32:32:8-p10:20:32:32:32:8-p11:20:32:32:32:8-p12:20:32:32:32:8-p13:20:32:32:32:8-p14:20:32:32:32:8-p15:20:32:32:32:8"
target triple = "aie2p"

%struct.ipd.custom_type.uint2_t = type { i2 }
%struct.ipd.custom_type.tm_byte_t = type { i8 }

@_anonymous7 = external global [3 x i32]
@_anonymous6 = external global [3 x i32]
@_anonymous5 = external global [3 x i32]
@_anonymous4 = external global [3 x i32]
@_anonymous3 = external global [3 x i32]
@_anonymous2 = external global [3 x i32]
@_anonymous1 = external global [3 x i32]
@_anonymous0 = external global [3 x i32]
@A_L2L1_0_7_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_0_7_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_0_6_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_0_6_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_0_5_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_0_5_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_0_4_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_0_4_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_0_3_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_0_3_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_0_2_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_0_2_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_0_1_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_0_1_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_0_0_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_0_0_cons_buff_0 = external global [32 x [64 x i8]]
@B_L2L1_0_cons_buff_1 = external global [64 x [128 x i8]]
@B_L2L1_0_cons_buff_0 = external global [64 x [128 x i8]]
@B_L2L1_1_cons_buff_1 = external global [64 x [128 x i8]]
@B_L2L1_1_cons_buff_0 = external global [64 x [128 x i8]]
@B_L2L1_2_cons_buff_1 = external global [64 x [128 x i8]]
@B_L2L1_2_cons_buff_0 = external global [64 x [128 x i8]]
@B_L2L1_3_cons_buff_1 = external global [64 x [128 x i8]]
@B_L2L1_3_cons_buff_0 = external global [64 x [128 x i8]]
@B_L2L1_4_cons_buff_1 = external global [64 x [128 x i8]]
@B_L2L1_4_cons_buff_0 = external global [64 x [128 x i8]]
@B_L2L1_5_cons_buff_1 = external global [64 x [128 x i8]]
@B_L2L1_5_cons_buff_0 = external global [64 x [128 x i8]]
@B_L2L1_6_cons_buff_1 = external global [64 x [128 x i8]]
@B_L2L1_6_cons_buff_0 = external global [64 x [128 x i8]]
@B_L2L1_7_cons_buff_1 = external global [64 x [128 x i8]]
@B_L2L1_7_cons_buff_0 = external global [64 x [128 x i8]]
@C_L1L2_0_0_buff_0 = external global [32 x [128 x i16]]
@C_L1L2_1_0_buff_0 = external global [32 x [128 x i16]]
@C_L1L2_2_0_buff_0 = external global [32 x [128 x i16]]
@C_L1L2_3_0_buff_0 = external global [32 x [128 x i16]]
@C_L1L2_4_0_buff_0 = external global [32 x [128 x i16]]
@C_L1L2_5_0_buff_0 = external global [32 x [128 x i16]]
@C_L1L2_6_0_buff_0 = external global [32 x [128 x i16]]
@C_L1L2_7_0_buff_0 = external global [32 x [128 x i16]]

define void @core_7_2() addrspace(0) {
  store i32 0, ptr @_anonymous7, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  br label %1

1:                                                ; preds = %128, %0
  %2 = phi i64 [ %133, %128 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %134

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_7_0_buff_0)
  br label %5

5:                                                ; preds = %17, %4
  %6 = phi i64 [ %31, %17 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 32
  br i1 %7, label %8, label %32

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %11, %10
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %14, label %15 [
    i32 0, label %15
    i32 1, label %16
  ]

15:                                               ; preds = %12, %12
  br label %17

16:                                               ; preds = %12
  br label %17

17:                                               ; preds = %16, %15
  %18 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %16 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %15 ]
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, 1
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_i8_i16(ptr %19, ptr %20, ptr @C_L1L2_7_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %21 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %22 = add i32 %21, 1
  %23 = icmp sge i32 %22, 2
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %27 = add i32 %26, 1
  %28 = icmp sge i32 %27, 2
  %29 = add i32 %26, -1
  %30 = select i1 %28, i32 %29, i32 %27
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %31 = add i64 %6, 1
  br label %5

32:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %33 = load i32, ptr @_anonymous7, align 4
  %34 = add i32 %33, 1
  %35 = icmp sge i32 %34, 1
  %36 = select i1 %35, i32 %33, i32 %34
  store i32 %36, ptr @_anonymous7, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_7_0_buff_0)
  br label %37

37:                                               ; preds = %49, %32
  %38 = phi i64 [ %63, %49 ], [ 0, %32 ]
  %39 = icmp slt i64 %38, 32
  br i1 %39, label %40, label %64

40:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %41 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  switch i32 %41, label %42 [
    i32 0, label %42
    i32 1, label %43
  ]

42:                                               ; preds = %40, %40
  br label %44

43:                                               ; preds = %40
  br label %44

44:                                               ; preds = %43, %42
  %45 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %43 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %42 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %46 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %46, label %47 [
    i32 0, label %47
    i32 1, label %48
  ]

47:                                               ; preds = %44, %44
  br label %49

48:                                               ; preds = %44
  br label %49

49:                                               ; preds = %48, %47
  %50 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %48 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %47 ]
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, 1
  %52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 1
  call addrspace(0) void @matmul_i8_i16(ptr %51, ptr %52, ptr @C_L1L2_7_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %53 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %54 = add i32 %53, 1
  %55 = icmp sge i32 %54, 2
  %56 = add i32 %53, -1
  %57 = select i1 %55, i32 %56, i32 %54
  store i32 %57, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %58 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %59 = add i32 %58, 1
  %60 = icmp sge i32 %59, 2
  %61 = add i32 %58, -1
  %62 = select i1 %60, i32 %61, i32 %59
  store i32 %62, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %63 = add i64 %38, 1
  br label %37

64:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %65 = load i32, ptr @_anonymous7, align 4
  %66 = add i32 %65, 1
  %67 = icmp sge i32 %66, 1
  %68 = select i1 %67, i32 %65, i32 %66
  store i32 %68, ptr @_anonymous7, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_7_0_buff_0)
  br label %69

69:                                               ; preds = %81, %64
  %70 = phi i64 [ %95, %81 ], [ 0, %64 ]
  %71 = icmp slt i64 %70, 32
  br i1 %71, label %72, label %96

72:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %73 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  switch i32 %73, label %74 [
    i32 0, label %74
    i32 1, label %75
  ]

74:                                               ; preds = %72, %72
  br label %76

75:                                               ; preds = %72
  br label %76

76:                                               ; preds = %75, %74
  %77 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %75 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %74 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %78 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %78, label %79 [
    i32 0, label %79
    i32 1, label %80
  ]

79:                                               ; preds = %76, %76
  br label %81

80:                                               ; preds = %76
  br label %81

81:                                               ; preds = %80, %79
  %82 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %80 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %79 ]
  %83 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 1
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, 1
  call addrspace(0) void @matmul_i8_i16(ptr %83, ptr %84, ptr @C_L1L2_7_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %85 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %86 = add i32 %85, 1
  %87 = icmp sge i32 %86, 2
  %88 = add i32 %85, -1
  %89 = select i1 %87, i32 %88, i32 %86
  store i32 %89, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %90 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %91 = add i32 %90, 1
  %92 = icmp sge i32 %91, 2
  %93 = add i32 %90, -1
  %94 = select i1 %92, i32 %93, i32 %91
  store i32 %94, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %95 = add i64 %70, 1
  br label %69

96:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %97 = load i32, ptr @_anonymous7, align 4
  %98 = add i32 %97, 1
  %99 = icmp sge i32 %98, 1
  %100 = select i1 %99, i32 %97, i32 %98
  store i32 %100, ptr @_anonymous7, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_7_0_buff_0)
  br label %101

101:                                              ; preds = %113, %96
  %102 = phi i64 [ %127, %113 ], [ 0, %96 ]
  %103 = icmp slt i64 %102, 32
  br i1 %103, label %104, label %128

104:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %105 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  switch i32 %105, label %106 [
    i32 0, label %106
    i32 1, label %107
  ]

106:                                              ; preds = %104, %104
  br label %108

107:                                              ; preds = %104
  br label %108

108:                                              ; preds = %107, %106
  %109 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %107 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %106 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %110 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %110, label %111 [
    i32 0, label %111
    i32 1, label %112
  ]

111:                                              ; preds = %108, %108
  br label %113

112:                                              ; preds = %108
  br label %113

113:                                              ; preds = %112, %111
  %114 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %112 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %111 ]
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, 1
  call addrspace(0) void @matmul_i8_i16(ptr %115, ptr %116, ptr @C_L1L2_7_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %117 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %118 = add i32 %117, 1
  %119 = icmp sge i32 %118, 2
  %120 = add i32 %117, -1
  %121 = select i1 %119, i32 %120, i32 %118
  store i32 %121, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %122 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %123 = add i32 %122, 1
  %124 = icmp sge i32 %123, 2
  %125 = add i32 %122, -1
  %126 = select i1 %124, i32 %125, i32 %123
  store i32 %126, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %127 = add i64 %102, 1
  br label %101

128:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %129 = load i32, ptr @_anonymous7, align 4
  %130 = add i32 %129, 1
  %131 = icmp sge i32 %130, 1
  %132 = select i1 %131, i32 %129, i32 %130
  store i32 %132, ptr @_anonymous7, align 4
  %133 = add i64 %2, 1
  br label %1

134:                                              ; preds = %1
  ret void
}

declare void @llvm.aie2p.acquire(i32, i32) addrspace(0)

declare void @zero_i16(ptr) addrspace(0)

declare void @matmul_i8_i16(ptr, ptr, ptr) addrspace(0)

declare void @llvm.aie2p.release(i32, i32) addrspace(0)

define void @core_6_2() addrspace(0) {
  store i32 0, ptr @_anonymous6, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  br label %1

1:                                                ; preds = %128, %0
  %2 = phi i64 [ %133, %128 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %134

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_6_0_buff_0)
  br label %5

5:                                                ; preds = %17, %4
  %6 = phi i64 [ %31, %17 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 32
  br i1 %7, label %8, label %32

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %11, %10
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %14, label %15 [
    i32 0, label %15
    i32 1, label %16
  ]

15:                                               ; preds = %12, %12
  br label %17

16:                                               ; preds = %12
  br label %17

17:                                               ; preds = %16, %15
  %18 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %16 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %15 ]
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, 1
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_i8_i16(ptr %19, ptr %20, ptr @C_L1L2_6_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %21 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %22 = add i32 %21, 1
  %23 = icmp sge i32 %22, 2
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %27 = add i32 %26, 1
  %28 = icmp sge i32 %27, 2
  %29 = add i32 %26, -1
  %30 = select i1 %28, i32 %29, i32 %27
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %31 = add i64 %6, 1
  br label %5

32:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %33 = load i32, ptr @_anonymous6, align 4
  %34 = add i32 %33, 1
  %35 = icmp sge i32 %34, 1
  %36 = select i1 %35, i32 %33, i32 %34
  store i32 %36, ptr @_anonymous6, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_6_0_buff_0)
  br label %37

37:                                               ; preds = %49, %32
  %38 = phi i64 [ %63, %49 ], [ 0, %32 ]
  %39 = icmp slt i64 %38, 32
  br i1 %39, label %40, label %64

40:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %41 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  switch i32 %41, label %42 [
    i32 0, label %42
    i32 1, label %43
  ]

42:                                               ; preds = %40, %40
  br label %44

43:                                               ; preds = %40
  br label %44

44:                                               ; preds = %43, %42
  %45 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %43 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %42 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %46 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %46, label %47 [
    i32 0, label %47
    i32 1, label %48
  ]

47:                                               ; preds = %44, %44
  br label %49

48:                                               ; preds = %44
  br label %49

49:                                               ; preds = %48, %47
  %50 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %48 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %47 ]
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, 1
  %52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 1
  call addrspace(0) void @matmul_i8_i16(ptr %51, ptr %52, ptr @C_L1L2_6_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %53 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %54 = add i32 %53, 1
  %55 = icmp sge i32 %54, 2
  %56 = add i32 %53, -1
  %57 = select i1 %55, i32 %56, i32 %54
  store i32 %57, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %58 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %59 = add i32 %58, 1
  %60 = icmp sge i32 %59, 2
  %61 = add i32 %58, -1
  %62 = select i1 %60, i32 %61, i32 %59
  store i32 %62, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %63 = add i64 %38, 1
  br label %37

64:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %65 = load i32, ptr @_anonymous6, align 4
  %66 = add i32 %65, 1
  %67 = icmp sge i32 %66, 1
  %68 = select i1 %67, i32 %65, i32 %66
  store i32 %68, ptr @_anonymous6, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_6_0_buff_0)
  br label %69

69:                                               ; preds = %81, %64
  %70 = phi i64 [ %95, %81 ], [ 0, %64 ]
  %71 = icmp slt i64 %70, 32
  br i1 %71, label %72, label %96

72:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %73 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  switch i32 %73, label %74 [
    i32 0, label %74
    i32 1, label %75
  ]

74:                                               ; preds = %72, %72
  br label %76

75:                                               ; preds = %72
  br label %76

76:                                               ; preds = %75, %74
  %77 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %75 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %74 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %78 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %78, label %79 [
    i32 0, label %79
    i32 1, label %80
  ]

79:                                               ; preds = %76, %76
  br label %81

80:                                               ; preds = %76
  br label %81

81:                                               ; preds = %80, %79
  %82 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %80 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %79 ]
  %83 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 1
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, 1
  call addrspace(0) void @matmul_i8_i16(ptr %83, ptr %84, ptr @C_L1L2_6_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %85 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %86 = add i32 %85, 1
  %87 = icmp sge i32 %86, 2
  %88 = add i32 %85, -1
  %89 = select i1 %87, i32 %88, i32 %86
  store i32 %89, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %90 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %91 = add i32 %90, 1
  %92 = icmp sge i32 %91, 2
  %93 = add i32 %90, -1
  %94 = select i1 %92, i32 %93, i32 %91
  store i32 %94, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %95 = add i64 %70, 1
  br label %69

96:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %97 = load i32, ptr @_anonymous6, align 4
  %98 = add i32 %97, 1
  %99 = icmp sge i32 %98, 1
  %100 = select i1 %99, i32 %97, i32 %98
  store i32 %100, ptr @_anonymous6, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_6_0_buff_0)
  br label %101

101:                                              ; preds = %113, %96
  %102 = phi i64 [ %127, %113 ], [ 0, %96 ]
  %103 = icmp slt i64 %102, 32
  br i1 %103, label %104, label %128

104:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %105 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  switch i32 %105, label %106 [
    i32 0, label %106
    i32 1, label %107
  ]

106:                                              ; preds = %104, %104
  br label %108

107:                                              ; preds = %104
  br label %108

108:                                              ; preds = %107, %106
  %109 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %107 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %106 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %110 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %110, label %111 [
    i32 0, label %111
    i32 1, label %112
  ]

111:                                              ; preds = %108, %108
  br label %113

112:                                              ; preds = %108
  br label %113

113:                                              ; preds = %112, %111
  %114 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %112 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %111 ]
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, 1
  call addrspace(0) void @matmul_i8_i16(ptr %115, ptr %116, ptr @C_L1L2_6_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %117 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %118 = add i32 %117, 1
  %119 = icmp sge i32 %118, 2
  %120 = add i32 %117, -1
  %121 = select i1 %119, i32 %120, i32 %118
  store i32 %121, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %122 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %123 = add i32 %122, 1
  %124 = icmp sge i32 %123, 2
  %125 = add i32 %122, -1
  %126 = select i1 %124, i32 %125, i32 %123
  store i32 %126, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %127 = add i64 %102, 1
  br label %101

128:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %129 = load i32, ptr @_anonymous6, align 4
  %130 = add i32 %129, 1
  %131 = icmp sge i32 %130, 1
  %132 = select i1 %131, i32 %129, i32 %130
  store i32 %132, ptr @_anonymous6, align 4
  %133 = add i64 %2, 1
  br label %1

134:                                              ; preds = %1
  ret void
}

define void @core_5_2() addrspace(0) {
  store i32 0, ptr @_anonymous5, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  br label %1

1:                                                ; preds = %128, %0
  %2 = phi i64 [ %133, %128 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %134

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_5_0_buff_0)
  br label %5

5:                                                ; preds = %17, %4
  %6 = phi i64 [ %31, %17 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 32
  br i1 %7, label %8, label %32

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %11, %10
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %14, label %15 [
    i32 0, label %15
    i32 1, label %16
  ]

15:                                               ; preds = %12, %12
  br label %17

16:                                               ; preds = %12
  br label %17

17:                                               ; preds = %16, %15
  %18 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %16 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %15 ]
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, 1
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_i8_i16(ptr %19, ptr %20, ptr @C_L1L2_5_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %21 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %22 = add i32 %21, 1
  %23 = icmp sge i32 %22, 2
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %27 = add i32 %26, 1
  %28 = icmp sge i32 %27, 2
  %29 = add i32 %26, -1
  %30 = select i1 %28, i32 %29, i32 %27
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %31 = add i64 %6, 1
  br label %5

32:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %33 = load i32, ptr @_anonymous5, align 4
  %34 = add i32 %33, 1
  %35 = icmp sge i32 %34, 1
  %36 = select i1 %35, i32 %33, i32 %34
  store i32 %36, ptr @_anonymous5, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_5_0_buff_0)
  br label %37

37:                                               ; preds = %49, %32
  %38 = phi i64 [ %63, %49 ], [ 0, %32 ]
  %39 = icmp slt i64 %38, 32
  br i1 %39, label %40, label %64

40:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %41 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  switch i32 %41, label %42 [
    i32 0, label %42
    i32 1, label %43
  ]

42:                                               ; preds = %40, %40
  br label %44

43:                                               ; preds = %40
  br label %44

44:                                               ; preds = %43, %42
  %45 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %43 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %42 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %46 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %46, label %47 [
    i32 0, label %47
    i32 1, label %48
  ]

47:                                               ; preds = %44, %44
  br label %49

48:                                               ; preds = %44
  br label %49

49:                                               ; preds = %48, %47
  %50 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %48 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %47 ]
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, 1
  %52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 1
  call addrspace(0) void @matmul_i8_i16(ptr %51, ptr %52, ptr @C_L1L2_5_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %53 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %54 = add i32 %53, 1
  %55 = icmp sge i32 %54, 2
  %56 = add i32 %53, -1
  %57 = select i1 %55, i32 %56, i32 %54
  store i32 %57, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %58 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %59 = add i32 %58, 1
  %60 = icmp sge i32 %59, 2
  %61 = add i32 %58, -1
  %62 = select i1 %60, i32 %61, i32 %59
  store i32 %62, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %63 = add i64 %38, 1
  br label %37

64:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %65 = load i32, ptr @_anonymous5, align 4
  %66 = add i32 %65, 1
  %67 = icmp sge i32 %66, 1
  %68 = select i1 %67, i32 %65, i32 %66
  store i32 %68, ptr @_anonymous5, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_5_0_buff_0)
  br label %69

69:                                               ; preds = %81, %64
  %70 = phi i64 [ %95, %81 ], [ 0, %64 ]
  %71 = icmp slt i64 %70, 32
  br i1 %71, label %72, label %96

72:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %73 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  switch i32 %73, label %74 [
    i32 0, label %74
    i32 1, label %75
  ]

74:                                               ; preds = %72, %72
  br label %76

75:                                               ; preds = %72
  br label %76

76:                                               ; preds = %75, %74
  %77 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %75 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %74 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %78 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %78, label %79 [
    i32 0, label %79
    i32 1, label %80
  ]

79:                                               ; preds = %76, %76
  br label %81

80:                                               ; preds = %76
  br label %81

81:                                               ; preds = %80, %79
  %82 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %80 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %79 ]
  %83 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 1
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, 1
  call addrspace(0) void @matmul_i8_i16(ptr %83, ptr %84, ptr @C_L1L2_5_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %85 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %86 = add i32 %85, 1
  %87 = icmp sge i32 %86, 2
  %88 = add i32 %85, -1
  %89 = select i1 %87, i32 %88, i32 %86
  store i32 %89, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %90 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %91 = add i32 %90, 1
  %92 = icmp sge i32 %91, 2
  %93 = add i32 %90, -1
  %94 = select i1 %92, i32 %93, i32 %91
  store i32 %94, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %95 = add i64 %70, 1
  br label %69

96:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %97 = load i32, ptr @_anonymous5, align 4
  %98 = add i32 %97, 1
  %99 = icmp sge i32 %98, 1
  %100 = select i1 %99, i32 %97, i32 %98
  store i32 %100, ptr @_anonymous5, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_5_0_buff_0)
  br label %101

101:                                              ; preds = %113, %96
  %102 = phi i64 [ %127, %113 ], [ 0, %96 ]
  %103 = icmp slt i64 %102, 32
  br i1 %103, label %104, label %128

104:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %105 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  switch i32 %105, label %106 [
    i32 0, label %106
    i32 1, label %107
  ]

106:                                              ; preds = %104, %104
  br label %108

107:                                              ; preds = %104
  br label %108

108:                                              ; preds = %107, %106
  %109 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %107 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %106 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %110 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %110, label %111 [
    i32 0, label %111
    i32 1, label %112
  ]

111:                                              ; preds = %108, %108
  br label %113

112:                                              ; preds = %108
  br label %113

113:                                              ; preds = %112, %111
  %114 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %112 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %111 ]
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, 1
  call addrspace(0) void @matmul_i8_i16(ptr %115, ptr %116, ptr @C_L1L2_5_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %117 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %118 = add i32 %117, 1
  %119 = icmp sge i32 %118, 2
  %120 = add i32 %117, -1
  %121 = select i1 %119, i32 %120, i32 %118
  store i32 %121, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %122 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %123 = add i32 %122, 1
  %124 = icmp sge i32 %123, 2
  %125 = add i32 %122, -1
  %126 = select i1 %124, i32 %125, i32 %123
  store i32 %126, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %127 = add i64 %102, 1
  br label %101

128:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %129 = load i32, ptr @_anonymous5, align 4
  %130 = add i32 %129, 1
  %131 = icmp sge i32 %130, 1
  %132 = select i1 %131, i32 %129, i32 %130
  store i32 %132, ptr @_anonymous5, align 4
  %133 = add i64 %2, 1
  br label %1

134:                                              ; preds = %1
  ret void
}

define void @core_4_2() addrspace(0) {
  store i32 0, ptr @_anonymous4, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  br label %1

1:                                                ; preds = %128, %0
  %2 = phi i64 [ %133, %128 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %134

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_4_0_buff_0)
  br label %5

5:                                                ; preds = %17, %4
  %6 = phi i64 [ %31, %17 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 32
  br i1 %7, label %8, label %32

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %11, %10
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %14, label %15 [
    i32 0, label %15
    i32 1, label %16
  ]

15:                                               ; preds = %12, %12
  br label %17

16:                                               ; preds = %12
  br label %17

17:                                               ; preds = %16, %15
  %18 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %16 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %15 ]
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, 1
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_i8_i16(ptr %19, ptr %20, ptr @C_L1L2_4_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %21 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %22 = add i32 %21, 1
  %23 = icmp sge i32 %22, 2
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %27 = add i32 %26, 1
  %28 = icmp sge i32 %27, 2
  %29 = add i32 %26, -1
  %30 = select i1 %28, i32 %29, i32 %27
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %31 = add i64 %6, 1
  br label %5

32:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %33 = load i32, ptr @_anonymous4, align 4
  %34 = add i32 %33, 1
  %35 = icmp sge i32 %34, 1
  %36 = select i1 %35, i32 %33, i32 %34
  store i32 %36, ptr @_anonymous4, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_4_0_buff_0)
  br label %37

37:                                               ; preds = %49, %32
  %38 = phi i64 [ %63, %49 ], [ 0, %32 ]
  %39 = icmp slt i64 %38, 32
  br i1 %39, label %40, label %64

40:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %41 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %41, label %42 [
    i32 0, label %42
    i32 1, label %43
  ]

42:                                               ; preds = %40, %40
  br label %44

43:                                               ; preds = %40
  br label %44

44:                                               ; preds = %43, %42
  %45 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %43 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %42 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %46 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %46, label %47 [
    i32 0, label %47
    i32 1, label %48
  ]

47:                                               ; preds = %44, %44
  br label %49

48:                                               ; preds = %44
  br label %49

49:                                               ; preds = %48, %47
  %50 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %48 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %47 ]
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, 1
  %52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 1
  call addrspace(0) void @matmul_i8_i16(ptr %51, ptr %52, ptr @C_L1L2_4_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %53 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %54 = add i32 %53, 1
  %55 = icmp sge i32 %54, 2
  %56 = add i32 %53, -1
  %57 = select i1 %55, i32 %56, i32 %54
  store i32 %57, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %58 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %59 = add i32 %58, 1
  %60 = icmp sge i32 %59, 2
  %61 = add i32 %58, -1
  %62 = select i1 %60, i32 %61, i32 %59
  store i32 %62, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %63 = add i64 %38, 1
  br label %37

64:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %65 = load i32, ptr @_anonymous4, align 4
  %66 = add i32 %65, 1
  %67 = icmp sge i32 %66, 1
  %68 = select i1 %67, i32 %65, i32 %66
  store i32 %68, ptr @_anonymous4, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_4_0_buff_0)
  br label %69

69:                                               ; preds = %81, %64
  %70 = phi i64 [ %95, %81 ], [ 0, %64 ]
  %71 = icmp slt i64 %70, 32
  br i1 %71, label %72, label %96

72:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %73 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %73, label %74 [
    i32 0, label %74
    i32 1, label %75
  ]

74:                                               ; preds = %72, %72
  br label %76

75:                                               ; preds = %72
  br label %76

76:                                               ; preds = %75, %74
  %77 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %75 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %74 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %78 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %78, label %79 [
    i32 0, label %79
    i32 1, label %80
  ]

79:                                               ; preds = %76, %76
  br label %81

80:                                               ; preds = %76
  br label %81

81:                                               ; preds = %80, %79
  %82 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %80 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %79 ]
  %83 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 1
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, 1
  call addrspace(0) void @matmul_i8_i16(ptr %83, ptr %84, ptr @C_L1L2_4_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %85 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %86 = add i32 %85, 1
  %87 = icmp sge i32 %86, 2
  %88 = add i32 %85, -1
  %89 = select i1 %87, i32 %88, i32 %86
  store i32 %89, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %90 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %91 = add i32 %90, 1
  %92 = icmp sge i32 %91, 2
  %93 = add i32 %90, -1
  %94 = select i1 %92, i32 %93, i32 %91
  store i32 %94, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %95 = add i64 %70, 1
  br label %69

96:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %97 = load i32, ptr @_anonymous4, align 4
  %98 = add i32 %97, 1
  %99 = icmp sge i32 %98, 1
  %100 = select i1 %99, i32 %97, i32 %98
  store i32 %100, ptr @_anonymous4, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_4_0_buff_0)
  br label %101

101:                                              ; preds = %113, %96
  %102 = phi i64 [ %127, %113 ], [ 0, %96 ]
  %103 = icmp slt i64 %102, 32
  br i1 %103, label %104, label %128

104:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %105 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %105, label %106 [
    i32 0, label %106
    i32 1, label %107
  ]

106:                                              ; preds = %104, %104
  br label %108

107:                                              ; preds = %104
  br label %108

108:                                              ; preds = %107, %106
  %109 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %107 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %106 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %110 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %110, label %111 [
    i32 0, label %111
    i32 1, label %112
  ]

111:                                              ; preds = %108, %108
  br label %113

112:                                              ; preds = %108
  br label %113

113:                                              ; preds = %112, %111
  %114 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %112 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %111 ]
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, 1
  call addrspace(0) void @matmul_i8_i16(ptr %115, ptr %116, ptr @C_L1L2_4_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %117 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %118 = add i32 %117, 1
  %119 = icmp sge i32 %118, 2
  %120 = add i32 %117, -1
  %121 = select i1 %119, i32 %120, i32 %118
  store i32 %121, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %122 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %123 = add i32 %122, 1
  %124 = icmp sge i32 %123, 2
  %125 = add i32 %122, -1
  %126 = select i1 %124, i32 %125, i32 %123
  store i32 %126, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %127 = add i64 %102, 1
  br label %101

128:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %129 = load i32, ptr @_anonymous4, align 4
  %130 = add i32 %129, 1
  %131 = icmp sge i32 %130, 1
  %132 = select i1 %131, i32 %129, i32 %130
  store i32 %132, ptr @_anonymous4, align 4
  %133 = add i64 %2, 1
  br label %1

134:                                              ; preds = %1
  ret void
}

define void @core_3_2() addrspace(0) {
  store i32 0, ptr @_anonymous3, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  br label %1

1:                                                ; preds = %128, %0
  %2 = phi i64 [ %133, %128 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %134

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_3_0_buff_0)
  br label %5

5:                                                ; preds = %17, %4
  %6 = phi i64 [ %31, %17 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 32
  br i1 %7, label %8, label %32

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %11, %10
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %14, label %15 [
    i32 0, label %15
    i32 1, label %16
  ]

15:                                               ; preds = %12, %12
  br label %17

16:                                               ; preds = %12
  br label %17

17:                                               ; preds = %16, %15
  %18 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %16 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %15 ]
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, 1
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_i8_i16(ptr %19, ptr %20, ptr @C_L1L2_3_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %21 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %22 = add i32 %21, 1
  %23 = icmp sge i32 %22, 2
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %27 = add i32 %26, 1
  %28 = icmp sge i32 %27, 2
  %29 = add i32 %26, -1
  %30 = select i1 %28, i32 %29, i32 %27
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %31 = add i64 %6, 1
  br label %5

32:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %33 = load i32, ptr @_anonymous3, align 4
  %34 = add i32 %33, 1
  %35 = icmp sge i32 %34, 1
  %36 = select i1 %35, i32 %33, i32 %34
  store i32 %36, ptr @_anonymous3, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_3_0_buff_0)
  br label %37

37:                                               ; preds = %49, %32
  %38 = phi i64 [ %63, %49 ], [ 0, %32 ]
  %39 = icmp slt i64 %38, 32
  br i1 %39, label %40, label %64

40:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %41 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  switch i32 %41, label %42 [
    i32 0, label %42
    i32 1, label %43
  ]

42:                                               ; preds = %40, %40
  br label %44

43:                                               ; preds = %40
  br label %44

44:                                               ; preds = %43, %42
  %45 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %43 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %42 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %46 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %46, label %47 [
    i32 0, label %47
    i32 1, label %48
  ]

47:                                               ; preds = %44, %44
  br label %49

48:                                               ; preds = %44
  br label %49

49:                                               ; preds = %48, %47
  %50 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %48 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %47 ]
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, 1
  %52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 1
  call addrspace(0) void @matmul_i8_i16(ptr %51, ptr %52, ptr @C_L1L2_3_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %53 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %54 = add i32 %53, 1
  %55 = icmp sge i32 %54, 2
  %56 = add i32 %53, -1
  %57 = select i1 %55, i32 %56, i32 %54
  store i32 %57, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %58 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %59 = add i32 %58, 1
  %60 = icmp sge i32 %59, 2
  %61 = add i32 %58, -1
  %62 = select i1 %60, i32 %61, i32 %59
  store i32 %62, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %63 = add i64 %38, 1
  br label %37

64:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %65 = load i32, ptr @_anonymous3, align 4
  %66 = add i32 %65, 1
  %67 = icmp sge i32 %66, 1
  %68 = select i1 %67, i32 %65, i32 %66
  store i32 %68, ptr @_anonymous3, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_3_0_buff_0)
  br label %69

69:                                               ; preds = %81, %64
  %70 = phi i64 [ %95, %81 ], [ 0, %64 ]
  %71 = icmp slt i64 %70, 32
  br i1 %71, label %72, label %96

72:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %73 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  switch i32 %73, label %74 [
    i32 0, label %74
    i32 1, label %75
  ]

74:                                               ; preds = %72, %72
  br label %76

75:                                               ; preds = %72
  br label %76

76:                                               ; preds = %75, %74
  %77 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %75 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %74 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %78 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %78, label %79 [
    i32 0, label %79
    i32 1, label %80
  ]

79:                                               ; preds = %76, %76
  br label %81

80:                                               ; preds = %76
  br label %81

81:                                               ; preds = %80, %79
  %82 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %80 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %79 ]
  %83 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 1
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, 1
  call addrspace(0) void @matmul_i8_i16(ptr %83, ptr %84, ptr @C_L1L2_3_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %85 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %86 = add i32 %85, 1
  %87 = icmp sge i32 %86, 2
  %88 = add i32 %85, -1
  %89 = select i1 %87, i32 %88, i32 %86
  store i32 %89, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %90 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %91 = add i32 %90, 1
  %92 = icmp sge i32 %91, 2
  %93 = add i32 %90, -1
  %94 = select i1 %92, i32 %93, i32 %91
  store i32 %94, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %95 = add i64 %70, 1
  br label %69

96:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %97 = load i32, ptr @_anonymous3, align 4
  %98 = add i32 %97, 1
  %99 = icmp sge i32 %98, 1
  %100 = select i1 %99, i32 %97, i32 %98
  store i32 %100, ptr @_anonymous3, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_3_0_buff_0)
  br label %101

101:                                              ; preds = %113, %96
  %102 = phi i64 [ %127, %113 ], [ 0, %96 ]
  %103 = icmp slt i64 %102, 32
  br i1 %103, label %104, label %128

104:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %105 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  switch i32 %105, label %106 [
    i32 0, label %106
    i32 1, label %107
  ]

106:                                              ; preds = %104, %104
  br label %108

107:                                              ; preds = %104
  br label %108

108:                                              ; preds = %107, %106
  %109 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %107 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %106 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %110 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %110, label %111 [
    i32 0, label %111
    i32 1, label %112
  ]

111:                                              ; preds = %108, %108
  br label %113

112:                                              ; preds = %108
  br label %113

113:                                              ; preds = %112, %111
  %114 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %112 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %111 ]
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, 1
  call addrspace(0) void @matmul_i8_i16(ptr %115, ptr %116, ptr @C_L1L2_3_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %117 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %118 = add i32 %117, 1
  %119 = icmp sge i32 %118, 2
  %120 = add i32 %117, -1
  %121 = select i1 %119, i32 %120, i32 %118
  store i32 %121, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %122 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %123 = add i32 %122, 1
  %124 = icmp sge i32 %123, 2
  %125 = add i32 %122, -1
  %126 = select i1 %124, i32 %125, i32 %123
  store i32 %126, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %127 = add i64 %102, 1
  br label %101

128:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %129 = load i32, ptr @_anonymous3, align 4
  %130 = add i32 %129, 1
  %131 = icmp sge i32 %130, 1
  %132 = select i1 %131, i32 %129, i32 %130
  store i32 %132, ptr @_anonymous3, align 4
  %133 = add i64 %2, 1
  br label %1

134:                                              ; preds = %1
  ret void
}

define void @core_2_2() addrspace(0) {
  store i32 0, ptr @_anonymous2, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  br label %1

1:                                                ; preds = %128, %0
  %2 = phi i64 [ %133, %128 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %134

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_2_0_buff_0)
  br label %5

5:                                                ; preds = %17, %4
  %6 = phi i64 [ %31, %17 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 32
  br i1 %7, label %8, label %32

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %11, %10
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %14, label %15 [
    i32 0, label %15
    i32 1, label %16
  ]

15:                                               ; preds = %12, %12
  br label %17

16:                                               ; preds = %12
  br label %17

17:                                               ; preds = %16, %15
  %18 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %16 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %15 ]
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, 1
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_i8_i16(ptr %19, ptr %20, ptr @C_L1L2_2_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %21 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %22 = add i32 %21, 1
  %23 = icmp sge i32 %22, 2
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %27 = add i32 %26, 1
  %28 = icmp sge i32 %27, 2
  %29 = add i32 %26, -1
  %30 = select i1 %28, i32 %29, i32 %27
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %31 = add i64 %6, 1
  br label %5

32:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %33 = load i32, ptr @_anonymous2, align 4
  %34 = add i32 %33, 1
  %35 = icmp sge i32 %34, 1
  %36 = select i1 %35, i32 %33, i32 %34
  store i32 %36, ptr @_anonymous2, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_2_0_buff_0)
  br label %37

37:                                               ; preds = %49, %32
  %38 = phi i64 [ %63, %49 ], [ 0, %32 ]
  %39 = icmp slt i64 %38, 32
  br i1 %39, label %40, label %64

40:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %41 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  switch i32 %41, label %42 [
    i32 0, label %42
    i32 1, label %43
  ]

42:                                               ; preds = %40, %40
  br label %44

43:                                               ; preds = %40
  br label %44

44:                                               ; preds = %43, %42
  %45 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %43 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %42 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %46 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %46, label %47 [
    i32 0, label %47
    i32 1, label %48
  ]

47:                                               ; preds = %44, %44
  br label %49

48:                                               ; preds = %44
  br label %49

49:                                               ; preds = %48, %47
  %50 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %48 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %47 ]
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, 1
  %52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 1
  call addrspace(0) void @matmul_i8_i16(ptr %51, ptr %52, ptr @C_L1L2_2_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %53 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %54 = add i32 %53, 1
  %55 = icmp sge i32 %54, 2
  %56 = add i32 %53, -1
  %57 = select i1 %55, i32 %56, i32 %54
  store i32 %57, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %58 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %59 = add i32 %58, 1
  %60 = icmp sge i32 %59, 2
  %61 = add i32 %58, -1
  %62 = select i1 %60, i32 %61, i32 %59
  store i32 %62, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %63 = add i64 %38, 1
  br label %37

64:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %65 = load i32, ptr @_anonymous2, align 4
  %66 = add i32 %65, 1
  %67 = icmp sge i32 %66, 1
  %68 = select i1 %67, i32 %65, i32 %66
  store i32 %68, ptr @_anonymous2, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_2_0_buff_0)
  br label %69

69:                                               ; preds = %81, %64
  %70 = phi i64 [ %95, %81 ], [ 0, %64 ]
  %71 = icmp slt i64 %70, 32
  br i1 %71, label %72, label %96

72:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %73 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  switch i32 %73, label %74 [
    i32 0, label %74
    i32 1, label %75
  ]

74:                                               ; preds = %72, %72
  br label %76

75:                                               ; preds = %72
  br label %76

76:                                               ; preds = %75, %74
  %77 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %75 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %74 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %78 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %78, label %79 [
    i32 0, label %79
    i32 1, label %80
  ]

79:                                               ; preds = %76, %76
  br label %81

80:                                               ; preds = %76
  br label %81

81:                                               ; preds = %80, %79
  %82 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %80 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %79 ]
  %83 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 1
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, 1
  call addrspace(0) void @matmul_i8_i16(ptr %83, ptr %84, ptr @C_L1L2_2_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %85 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %86 = add i32 %85, 1
  %87 = icmp sge i32 %86, 2
  %88 = add i32 %85, -1
  %89 = select i1 %87, i32 %88, i32 %86
  store i32 %89, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %90 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %91 = add i32 %90, 1
  %92 = icmp sge i32 %91, 2
  %93 = add i32 %90, -1
  %94 = select i1 %92, i32 %93, i32 %91
  store i32 %94, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %95 = add i64 %70, 1
  br label %69

96:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %97 = load i32, ptr @_anonymous2, align 4
  %98 = add i32 %97, 1
  %99 = icmp sge i32 %98, 1
  %100 = select i1 %99, i32 %97, i32 %98
  store i32 %100, ptr @_anonymous2, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_2_0_buff_0)
  br label %101

101:                                              ; preds = %113, %96
  %102 = phi i64 [ %127, %113 ], [ 0, %96 ]
  %103 = icmp slt i64 %102, 32
  br i1 %103, label %104, label %128

104:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %105 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  switch i32 %105, label %106 [
    i32 0, label %106
    i32 1, label %107
  ]

106:                                              ; preds = %104, %104
  br label %108

107:                                              ; preds = %104
  br label %108

108:                                              ; preds = %107, %106
  %109 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %107 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %106 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %110 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %110, label %111 [
    i32 0, label %111
    i32 1, label %112
  ]

111:                                              ; preds = %108, %108
  br label %113

112:                                              ; preds = %108
  br label %113

113:                                              ; preds = %112, %111
  %114 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %112 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %111 ]
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, 1
  call addrspace(0) void @matmul_i8_i16(ptr %115, ptr %116, ptr @C_L1L2_2_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %117 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %118 = add i32 %117, 1
  %119 = icmp sge i32 %118, 2
  %120 = add i32 %117, -1
  %121 = select i1 %119, i32 %120, i32 %118
  store i32 %121, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %122 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %123 = add i32 %122, 1
  %124 = icmp sge i32 %123, 2
  %125 = add i32 %122, -1
  %126 = select i1 %124, i32 %125, i32 %123
  store i32 %126, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %127 = add i64 %102, 1
  br label %101

128:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %129 = load i32, ptr @_anonymous2, align 4
  %130 = add i32 %129, 1
  %131 = icmp sge i32 %130, 1
  %132 = select i1 %131, i32 %129, i32 %130
  store i32 %132, ptr @_anonymous2, align 4
  %133 = add i64 %2, 1
  br label %1

134:                                              ; preds = %1
  ret void
}

define void @core_1_2() addrspace(0) {
  store i32 0, ptr @_anonymous1, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  br label %1

1:                                                ; preds = %128, %0
  %2 = phi i64 [ %133, %128 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %134

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_1_0_buff_0)
  br label %5

5:                                                ; preds = %17, %4
  %6 = phi i64 [ %31, %17 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 32
  br i1 %7, label %8, label %32

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %11, %10
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %14, label %15 [
    i32 0, label %15
    i32 1, label %16
  ]

15:                                               ; preds = %12, %12
  br label %17

16:                                               ; preds = %12
  br label %17

17:                                               ; preds = %16, %15
  %18 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %16 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %15 ]
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, 1
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_i8_i16(ptr %19, ptr %20, ptr @C_L1L2_1_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %21 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %22 = add i32 %21, 1
  %23 = icmp sge i32 %22, 2
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %27 = add i32 %26, 1
  %28 = icmp sge i32 %27, 2
  %29 = add i32 %26, -1
  %30 = select i1 %28, i32 %29, i32 %27
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %31 = add i64 %6, 1
  br label %5

32:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %33 = load i32, ptr @_anonymous1, align 4
  %34 = add i32 %33, 1
  %35 = icmp sge i32 %34, 1
  %36 = select i1 %35, i32 %33, i32 %34
  store i32 %36, ptr @_anonymous1, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_1_0_buff_0)
  br label %37

37:                                               ; preds = %49, %32
  %38 = phi i64 [ %63, %49 ], [ 0, %32 ]
  %39 = icmp slt i64 %38, 32
  br i1 %39, label %40, label %64

40:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %41 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  switch i32 %41, label %42 [
    i32 0, label %42
    i32 1, label %43
  ]

42:                                               ; preds = %40, %40
  br label %44

43:                                               ; preds = %40
  br label %44

44:                                               ; preds = %43, %42
  %45 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %43 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %42 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %46 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %46, label %47 [
    i32 0, label %47
    i32 1, label %48
  ]

47:                                               ; preds = %44, %44
  br label %49

48:                                               ; preds = %44
  br label %49

49:                                               ; preds = %48, %47
  %50 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %48 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %47 ]
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, 1
  %52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 1
  call addrspace(0) void @matmul_i8_i16(ptr %51, ptr %52, ptr @C_L1L2_1_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %53 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %54 = add i32 %53, 1
  %55 = icmp sge i32 %54, 2
  %56 = add i32 %53, -1
  %57 = select i1 %55, i32 %56, i32 %54
  store i32 %57, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %58 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %59 = add i32 %58, 1
  %60 = icmp sge i32 %59, 2
  %61 = add i32 %58, -1
  %62 = select i1 %60, i32 %61, i32 %59
  store i32 %62, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %63 = add i64 %38, 1
  br label %37

64:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %65 = load i32, ptr @_anonymous1, align 4
  %66 = add i32 %65, 1
  %67 = icmp sge i32 %66, 1
  %68 = select i1 %67, i32 %65, i32 %66
  store i32 %68, ptr @_anonymous1, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_1_0_buff_0)
  br label %69

69:                                               ; preds = %81, %64
  %70 = phi i64 [ %95, %81 ], [ 0, %64 ]
  %71 = icmp slt i64 %70, 32
  br i1 %71, label %72, label %96

72:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %73 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  switch i32 %73, label %74 [
    i32 0, label %74
    i32 1, label %75
  ]

74:                                               ; preds = %72, %72
  br label %76

75:                                               ; preds = %72
  br label %76

76:                                               ; preds = %75, %74
  %77 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %75 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %74 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %78 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %78, label %79 [
    i32 0, label %79
    i32 1, label %80
  ]

79:                                               ; preds = %76, %76
  br label %81

80:                                               ; preds = %76
  br label %81

81:                                               ; preds = %80, %79
  %82 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %80 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %79 ]
  %83 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 1
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, 1
  call addrspace(0) void @matmul_i8_i16(ptr %83, ptr %84, ptr @C_L1L2_1_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %85 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %86 = add i32 %85, 1
  %87 = icmp sge i32 %86, 2
  %88 = add i32 %85, -1
  %89 = select i1 %87, i32 %88, i32 %86
  store i32 %89, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %90 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %91 = add i32 %90, 1
  %92 = icmp sge i32 %91, 2
  %93 = add i32 %90, -1
  %94 = select i1 %92, i32 %93, i32 %91
  store i32 %94, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %95 = add i64 %70, 1
  br label %69

96:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %97 = load i32, ptr @_anonymous1, align 4
  %98 = add i32 %97, 1
  %99 = icmp sge i32 %98, 1
  %100 = select i1 %99, i32 %97, i32 %98
  store i32 %100, ptr @_anonymous1, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_1_0_buff_0)
  br label %101

101:                                              ; preds = %113, %96
  %102 = phi i64 [ %127, %113 ], [ 0, %96 ]
  %103 = icmp slt i64 %102, 32
  br i1 %103, label %104, label %128

104:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %105 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  switch i32 %105, label %106 [
    i32 0, label %106
    i32 1, label %107
  ]

106:                                              ; preds = %104, %104
  br label %108

107:                                              ; preds = %104
  br label %108

108:                                              ; preds = %107, %106
  %109 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %107 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %106 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %110 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %110, label %111 [
    i32 0, label %111
    i32 1, label %112
  ]

111:                                              ; preds = %108, %108
  br label %113

112:                                              ; preds = %108
  br label %113

113:                                              ; preds = %112, %111
  %114 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %112 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %111 ]
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, 1
  call addrspace(0) void @matmul_i8_i16(ptr %115, ptr %116, ptr @C_L1L2_1_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %117 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %118 = add i32 %117, 1
  %119 = icmp sge i32 %118, 2
  %120 = add i32 %117, -1
  %121 = select i1 %119, i32 %120, i32 %118
  store i32 %121, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %122 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %123 = add i32 %122, 1
  %124 = icmp sge i32 %123, 2
  %125 = add i32 %122, -1
  %126 = select i1 %124, i32 %125, i32 %123
  store i32 %126, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %127 = add i64 %102, 1
  br label %101

128:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %129 = load i32, ptr @_anonymous1, align 4
  %130 = add i32 %129, 1
  %131 = icmp sge i32 %130, 1
  %132 = select i1 %131, i32 %129, i32 %130
  store i32 %132, ptr @_anonymous1, align 4
  %133 = add i64 %2, 1
  br label %1

134:                                              ; preds = %1
  ret void
}

define void @core_0_2() addrspace(0) {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  br label %1

1:                                                ; preds = %128, %0
  %2 = phi i64 [ %133, %128 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %134

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_0_0_buff_0)
  br label %5

5:                                                ; preds = %17, %4
  %6 = phi i64 [ %31, %17 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 32
  br i1 %7, label %8, label %32

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %11, %10
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %14, label %15 [
    i32 0, label %15
    i32 1, label %16
  ]

15:                                               ; preds = %12, %12
  br label %17

16:                                               ; preds = %12
  br label %17

17:                                               ; preds = %16, %15
  %18 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %16 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %15 ]
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, 1
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_i8_i16(ptr %19, ptr %20, ptr @C_L1L2_0_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %21 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %22 = add i32 %21, 1
  %23 = icmp sge i32 %22, 2
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %27 = add i32 %26, 1
  %28 = icmp sge i32 %27, 2
  %29 = add i32 %26, -1
  %30 = select i1 %28, i32 %29, i32 %27
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %31 = add i64 %6, 1
  br label %5

32:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %33 = load i32, ptr @_anonymous0, align 4
  %34 = add i32 %33, 1
  %35 = icmp sge i32 %34, 1
  %36 = select i1 %35, i32 %33, i32 %34
  store i32 %36, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_0_0_buff_0)
  br label %37

37:                                               ; preds = %49, %32
  %38 = phi i64 [ %63, %49 ], [ 0, %32 ]
  %39 = icmp slt i64 %38, 32
  br i1 %39, label %40, label %64

40:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %41 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %41, label %42 [
    i32 0, label %42
    i32 1, label %43
  ]

42:                                               ; preds = %40, %40
  br label %44

43:                                               ; preds = %40
  br label %44

44:                                               ; preds = %43, %42
  %45 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %43 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %42 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %46 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %46, label %47 [
    i32 0, label %47
    i32 1, label %48
  ]

47:                                               ; preds = %44, %44
  br label %49

48:                                               ; preds = %44
  br label %49

49:                                               ; preds = %48, %47
  %50 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %48 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %47 ]
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, 1
  %52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 1
  call addrspace(0) void @matmul_i8_i16(ptr %51, ptr %52, ptr @C_L1L2_0_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %53 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %54 = add i32 %53, 1
  %55 = icmp sge i32 %54, 2
  %56 = add i32 %53, -1
  %57 = select i1 %55, i32 %56, i32 %54
  store i32 %57, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %58 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %59 = add i32 %58, 1
  %60 = icmp sge i32 %59, 2
  %61 = add i32 %58, -1
  %62 = select i1 %60, i32 %61, i32 %59
  store i32 %62, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %63 = add i64 %38, 1
  br label %37

64:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %65 = load i32, ptr @_anonymous0, align 4
  %66 = add i32 %65, 1
  %67 = icmp sge i32 %66, 1
  %68 = select i1 %67, i32 %65, i32 %66
  store i32 %68, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_0_0_buff_0)
  br label %69

69:                                               ; preds = %81, %64
  %70 = phi i64 [ %95, %81 ], [ 0, %64 ]
  %71 = icmp slt i64 %70, 32
  br i1 %71, label %72, label %96

72:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %73 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %73, label %74 [
    i32 0, label %74
    i32 1, label %75
  ]

74:                                               ; preds = %72, %72
  br label %76

75:                                               ; preds = %72
  br label %76

76:                                               ; preds = %75, %74
  %77 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %75 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %74 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %78 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %78, label %79 [
    i32 0, label %79
    i32 1, label %80
  ]

79:                                               ; preds = %76, %76
  br label %81

80:                                               ; preds = %76
  br label %81

81:                                               ; preds = %80, %79
  %82 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %80 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %79 ]
  %83 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 1
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, 1
  call addrspace(0) void @matmul_i8_i16(ptr %83, ptr %84, ptr @C_L1L2_0_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %85 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %86 = add i32 %85, 1
  %87 = icmp sge i32 %86, 2
  %88 = add i32 %85, -1
  %89 = select i1 %87, i32 %88, i32 %86
  store i32 %89, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %90 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %91 = add i32 %90, 1
  %92 = icmp sge i32 %91, 2
  %93 = add i32 %90, -1
  %94 = select i1 %92, i32 %93, i32 %91
  store i32 %94, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %95 = add i64 %70, 1
  br label %69

96:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %97 = load i32, ptr @_anonymous0, align 4
  %98 = add i32 %97, 1
  %99 = icmp sge i32 %98, 1
  %100 = select i1 %99, i32 %97, i32 %98
  store i32 %100, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_0_0_buff_0)
  br label %101

101:                                              ; preds = %113, %96
  %102 = phi i64 [ %127, %113 ], [ 0, %96 ]
  %103 = icmp slt i64 %102, 32
  br i1 %103, label %104, label %128

104:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %105 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %105, label %106 [
    i32 0, label %106
    i32 1, label %107
  ]

106:                                              ; preds = %104, %104
  br label %108

107:                                              ; preds = %104
  br label %108

108:                                              ; preds = %107, %106
  %109 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %107 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %106 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %110 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %110, label %111 [
    i32 0, label %111
    i32 1, label %112
  ]

111:                                              ; preds = %108, %108
  br label %113

112:                                              ; preds = %108
  br label %113

113:                                              ; preds = %112, %111
  %114 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %112 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %111 ]
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, 1
  call addrspace(0) void @matmul_i8_i16(ptr %115, ptr %116, ptr @C_L1L2_0_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %117 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %118 = add i32 %117, 1
  %119 = icmp sge i32 %118, 2
  %120 = add i32 %117, -1
  %121 = select i1 %119, i32 %120, i32 %118
  store i32 %121, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %122 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %123 = add i32 %122, 1
  %124 = icmp sge i32 %123, 2
  %125 = add i32 %122, -1
  %126 = select i1 %124, i32 %125, i32 %123
  store i32 %126, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %127 = add i64 %102, 1
  br label %101

128:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %129 = load i32, ptr @_anonymous0, align 4
  %130 = add i32 %129, 1
  %131 = icmp sge i32 %130, 1
  %132 = select i1 %131, i32 %129, i32 %130
  store i32 %132, ptr @_anonymous0, align 4
  %133 = add i64 %2, 1
  br label %1

134:                                              ; preds = %1
  ret void
}

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2p___acquire(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext %0, i32 zeroext %1) #4
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: nounwind willreturn
declare void @llvm.chess_memory_fence() addrspace(1) #1

; Function Attrs: nounwind memory(inaccessiblemem: readwrite)
declare dso_local void @_Z25chess_separator_schedulerv() local_unnamed_addr addrspace(1) #2

; Function Attrs: nounwind memory(inaccessiblemem: readwrite)
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext, i32 zeroext) local_unnamed_addr addrspace(1) #2

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2p___release(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext %0, i32 signext %1) #4
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: nounwind memory(inaccessiblemem: readwrite)
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext, i32 signext) local_unnamed_addr addrspace(1) #2

; Function Attrs: mustprogress nounwind memory(inaccessiblemem: readwrite)
define dso_local void @llvm___aie___event0() local_unnamed_addr addrspace(1) #3 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t zeroinitializer) #4
  ret void
}

; Function Attrs: nounwind memory(inaccessiblemem: readwrite)
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t) local_unnamed_addr addrspace(1) #2

; Function Attrs: mustprogress nounwind memory(inaccessiblemem: readwrite)
define dso_local void @llvm___aie___event1() local_unnamed_addr addrspace(1) #3 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t { i2 1 }) #4
  ret void
}

attributes #0 = { mustprogress nounwind "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind memory(inaccessiblemem: readwrite) "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { mustprogress nounwind memory(inaccessiblemem: readwrite) "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind memory(inaccessiblemem: readwrite) "no-builtin-memcpy" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.linker.options = !{}
!llvm.chess.memory-units = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 0, i8 undef}
!4 = !{i32 2, i8 undef}
!5 = !{i32 3, i8 undef}
!6 = !{i32 4, i8 undef}
!7 = !{i32 5, i8 undef}
!8 = !{i32 6, i8 undef}
!9 = !{i32 7, i8 undef}
!10 = !{i32 8, i8 undef}
!11 = !{i32 9, i8 undef}
!12 = !{i32 10, i8 undef}
!13 = !{i32 11, i8 undef}
!14 = !{i32 12, i8 undef}
!15 = !{i32 13, i8 undef}
!16 = !{i32 14, i8 undef}
!17 = !{i32 15, %struct.ipd.custom_type.tm_byte_t undef}
!18 = !{!"clang version 18.1.6 (/u/sgasip/ipd/repositories/llvm_ipd fe2f26b1fab3e45d6b25a43594b7f26e123552bd)"}
