; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-p:32:32:32:32:8-s0:512:512-a0:8:8-S512-n32:64-P1-p0:20:32:32:32:8-p1:20:32:32:32:8-p2:20:32:32:32:8-p3:20:32:32:32:8-p4:20:32:32:32:8-p5:20:32:32:32:8-p6:20:32:32:32:8-p7:20:32:32:32:8-p8:20:32:32:32:8-p9:20:32:32:32:8-p10:20:32:32:32:8-p11:20:32:32:32:8-p12:20:32:32:32:8-p13:20:32:32:32:8-p14:20:32:32:32:8-p15:20:32:32:32:8"
target triple = "aie2p"

%struct.ipd.custom_type.uint2_t = type { i2 }
%struct.ipd.custom_type.tm_byte_t = type { i8 }

@_anonymous0 = external global [3 x i32]
@A_L2L1_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_cons_buff_0 = external global [32 x [64 x i8]]
@B_L2L1_cons_buff_1 = external global [64 x [128 x i8]]
@B_L2L1_cons_buff_0 = external global [64 x [128 x i8]]
@C_L1L2_buff_0 = external global [32 x [128 x i16]]

define void @core_0_2() addrspace(0) {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  br label %1

1:                                                ; preds = %1024, %0
  %2 = phi i64 [ %1029, %1024 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %1030

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %5

5:                                                ; preds = %17, %4
  %6 = phi i64 [ %31, %17 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %32

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
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
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %10 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
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
  %18 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %16 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %15 ]
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %19, ptr %20, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %21 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %22 = add i32 %21, 1
  %23 = icmp sge i32 %22, 2
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %27 = add i32 %26, 1
  %28 = icmp sge i32 %27, 2
  %29 = add i32 %26, -1
  %30 = select i1 %28, i32 %29, i32 %27
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
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
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %37

37:                                               ; preds = %49, %32
  %38 = phi i64 [ %63, %49 ], [ 0, %32 ]
  %39 = icmp slt i64 %38, 16
  br i1 %39, label %40, label %64

40:                                               ; preds = %37
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
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
  %45 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %43 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %42 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
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
  %50 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %48 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %47 ]
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, 1
  %52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %51, ptr %52, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %53 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %54 = add i32 %53, 1
  %55 = icmp sge i32 %54, 2
  %56 = add i32 %53, -1
  %57 = select i1 %55, i32 %56, i32 %54
  store i32 %57, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %58 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %59 = add i32 %58, 1
  %60 = icmp sge i32 %59, 2
  %61 = add i32 %58, -1
  %62 = select i1 %60, i32 %61, i32 %59
  store i32 %62, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
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
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %69

69:                                               ; preds = %81, %64
  %70 = phi i64 [ %95, %81 ], [ 0, %64 ]
  %71 = icmp slt i64 %70, 16
  br i1 %71, label %72, label %96

72:                                               ; preds = %69
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
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
  %77 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %75 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %74 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
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
  %82 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %80 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %79 ]
  %83 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, 1
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %83, ptr %84, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %85 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %86 = add i32 %85, 1
  %87 = icmp sge i32 %86, 2
  %88 = add i32 %85, -1
  %89 = select i1 %87, i32 %88, i32 %86
  store i32 %89, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %90 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %91 = add i32 %90, 1
  %92 = icmp sge i32 %91, 2
  %93 = add i32 %90, -1
  %94 = select i1 %92, i32 %93, i32 %91
  store i32 %94, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
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
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %101

101:                                              ; preds = %113, %96
  %102 = phi i64 [ %127, %113 ], [ 0, %96 ]
  %103 = icmp slt i64 %102, 16
  br i1 %103, label %104, label %128

104:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
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
  %109 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %107 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %106 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
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
  %114 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %112 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %111 ]
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, 1
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %115, ptr %116, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %117 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %118 = add i32 %117, 1
  %119 = icmp sge i32 %118, 2
  %120 = add i32 %117, -1
  %121 = select i1 %119, i32 %120, i32 %118
  store i32 %121, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %122 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %123 = add i32 %122, 1
  %124 = icmp sge i32 %123, 2
  %125 = add i32 %122, -1
  %126 = select i1 %124, i32 %125, i32 %123
  store i32 %126, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %127 = add i64 %102, 1
  br label %101

128:                                              ; preds = %101
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %129 = load i32, ptr @_anonymous0, align 4
  %130 = add i32 %129, 1
  %131 = icmp sge i32 %130, 1
  %132 = select i1 %131, i32 %129, i32 %130
  store i32 %132, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %133

133:                                              ; preds = %145, %128
  %134 = phi i64 [ %159, %145 ], [ 0, %128 ]
  %135 = icmp slt i64 %134, 16
  br i1 %135, label %136, label %160

136:                                              ; preds = %133
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %137 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %137, label %138 [
    i32 0, label %138
    i32 1, label %139
  ]

138:                                              ; preds = %136, %136
  br label %140

139:                                              ; preds = %136
  br label %140

140:                                              ; preds = %139, %138
  %141 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %139 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %138 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %142 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %142, label %143 [
    i32 0, label %143
    i32 1, label %144
  ]

143:                                              ; preds = %140, %140
  br label %145

144:                                              ; preds = %140
  br label %145

145:                                              ; preds = %144, %143
  %146 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %144 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %143 ]
  %147 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %141, 1
  %148 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %146, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %147, ptr %148, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %149 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %150 = add i32 %149, 1
  %151 = icmp sge i32 %150, 2
  %152 = add i32 %149, -1
  %153 = select i1 %151, i32 %152, i32 %150
  store i32 %153, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %154 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %155 = add i32 %154, 1
  %156 = icmp sge i32 %155, 2
  %157 = add i32 %154, -1
  %158 = select i1 %156, i32 %157, i32 %155
  store i32 %158, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %159 = add i64 %134, 1
  br label %133

160:                                              ; preds = %133
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %161 = load i32, ptr @_anonymous0, align 4
  %162 = add i32 %161, 1
  %163 = icmp sge i32 %162, 1
  %164 = select i1 %163, i32 %161, i32 %162
  store i32 %164, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %165

165:                                              ; preds = %177, %160
  %166 = phi i64 [ %191, %177 ], [ 0, %160 ]
  %167 = icmp slt i64 %166, 16
  br i1 %167, label %168, label %192

168:                                              ; preds = %165
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %169 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %169, label %170 [
    i32 0, label %170
    i32 1, label %171
  ]

170:                                              ; preds = %168, %168
  br label %172

171:                                              ; preds = %168
  br label %172

172:                                              ; preds = %171, %170
  %173 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %171 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %170 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %174 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %174, label %175 [
    i32 0, label %175
    i32 1, label %176
  ]

175:                                              ; preds = %172, %172
  br label %177

176:                                              ; preds = %172
  br label %177

177:                                              ; preds = %176, %175
  %178 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %176 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %175 ]
  %179 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %173, 1
  %180 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %179, ptr %180, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %181 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %182 = add i32 %181, 1
  %183 = icmp sge i32 %182, 2
  %184 = add i32 %181, -1
  %185 = select i1 %183, i32 %184, i32 %182
  store i32 %185, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %186 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %187 = add i32 %186, 1
  %188 = icmp sge i32 %187, 2
  %189 = add i32 %186, -1
  %190 = select i1 %188, i32 %189, i32 %187
  store i32 %190, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %191 = add i64 %166, 1
  br label %165

192:                                              ; preds = %165
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %193 = load i32, ptr @_anonymous0, align 4
  %194 = add i32 %193, 1
  %195 = icmp sge i32 %194, 1
  %196 = select i1 %195, i32 %193, i32 %194
  store i32 %196, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %197

197:                                              ; preds = %209, %192
  %198 = phi i64 [ %223, %209 ], [ 0, %192 ]
  %199 = icmp slt i64 %198, 16
  br i1 %199, label %200, label %224

200:                                              ; preds = %197
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %201 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %201, label %202 [
    i32 0, label %202
    i32 1, label %203
  ]

202:                                              ; preds = %200, %200
  br label %204

203:                                              ; preds = %200
  br label %204

204:                                              ; preds = %203, %202
  %205 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %203 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %202 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %206 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %206, label %207 [
    i32 0, label %207
    i32 1, label %208
  ]

207:                                              ; preds = %204, %204
  br label %209

208:                                              ; preds = %204
  br label %209

209:                                              ; preds = %208, %207
  %210 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %208 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %207 ]
  %211 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %205, 1
  %212 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %210, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %211, ptr %212, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %213 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %214 = add i32 %213, 1
  %215 = icmp sge i32 %214, 2
  %216 = add i32 %213, -1
  %217 = select i1 %215, i32 %216, i32 %214
  store i32 %217, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %218 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %219 = add i32 %218, 1
  %220 = icmp sge i32 %219, 2
  %221 = add i32 %218, -1
  %222 = select i1 %220, i32 %221, i32 %219
  store i32 %222, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %223 = add i64 %198, 1
  br label %197

224:                                              ; preds = %197
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %225 = load i32, ptr @_anonymous0, align 4
  %226 = add i32 %225, 1
  %227 = icmp sge i32 %226, 1
  %228 = select i1 %227, i32 %225, i32 %226
  store i32 %228, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %229

229:                                              ; preds = %241, %224
  %230 = phi i64 [ %255, %241 ], [ 0, %224 ]
  %231 = icmp slt i64 %230, 16
  br i1 %231, label %232, label %256

232:                                              ; preds = %229
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %233 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %233, label %234 [
    i32 0, label %234
    i32 1, label %235
  ]

234:                                              ; preds = %232, %232
  br label %236

235:                                              ; preds = %232
  br label %236

236:                                              ; preds = %235, %234
  %237 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %235 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %234 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %238 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %238, label %239 [
    i32 0, label %239
    i32 1, label %240
  ]

239:                                              ; preds = %236, %236
  br label %241

240:                                              ; preds = %236
  br label %241

241:                                              ; preds = %240, %239
  %242 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %240 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %239 ]
  %243 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %237, 1
  %244 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %242, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %243, ptr %244, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %245 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %246 = add i32 %245, 1
  %247 = icmp sge i32 %246, 2
  %248 = add i32 %245, -1
  %249 = select i1 %247, i32 %248, i32 %246
  store i32 %249, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %250 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %251 = add i32 %250, 1
  %252 = icmp sge i32 %251, 2
  %253 = add i32 %250, -1
  %254 = select i1 %252, i32 %253, i32 %251
  store i32 %254, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %255 = add i64 %230, 1
  br label %229

256:                                              ; preds = %229
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %257 = load i32, ptr @_anonymous0, align 4
  %258 = add i32 %257, 1
  %259 = icmp sge i32 %258, 1
  %260 = select i1 %259, i32 %257, i32 %258
  store i32 %260, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %261

261:                                              ; preds = %273, %256
  %262 = phi i64 [ %287, %273 ], [ 0, %256 ]
  %263 = icmp slt i64 %262, 16
  br i1 %263, label %264, label %288

264:                                              ; preds = %261
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %265 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %265, label %266 [
    i32 0, label %266
    i32 1, label %267
  ]

266:                                              ; preds = %264, %264
  br label %268

267:                                              ; preds = %264
  br label %268

268:                                              ; preds = %267, %266
  %269 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %267 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %266 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %270 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %270, label %271 [
    i32 0, label %271
    i32 1, label %272
  ]

271:                                              ; preds = %268, %268
  br label %273

272:                                              ; preds = %268
  br label %273

273:                                              ; preds = %272, %271
  %274 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %272 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %271 ]
  %275 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %269, 1
  %276 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %274, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %275, ptr %276, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %277 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %278 = add i32 %277, 1
  %279 = icmp sge i32 %278, 2
  %280 = add i32 %277, -1
  %281 = select i1 %279, i32 %280, i32 %278
  store i32 %281, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %282 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %283 = add i32 %282, 1
  %284 = icmp sge i32 %283, 2
  %285 = add i32 %282, -1
  %286 = select i1 %284, i32 %285, i32 %283
  store i32 %286, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %287 = add i64 %262, 1
  br label %261

288:                                              ; preds = %261
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %289 = load i32, ptr @_anonymous0, align 4
  %290 = add i32 %289, 1
  %291 = icmp sge i32 %290, 1
  %292 = select i1 %291, i32 %289, i32 %290
  store i32 %292, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %293

293:                                              ; preds = %305, %288
  %294 = phi i64 [ %319, %305 ], [ 0, %288 ]
  %295 = icmp slt i64 %294, 16
  br i1 %295, label %296, label %320

296:                                              ; preds = %293
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %297 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %297, label %298 [
    i32 0, label %298
    i32 1, label %299
  ]

298:                                              ; preds = %296, %296
  br label %300

299:                                              ; preds = %296
  br label %300

300:                                              ; preds = %299, %298
  %301 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %299 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %298 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %302 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %302, label %303 [
    i32 0, label %303
    i32 1, label %304
  ]

303:                                              ; preds = %300, %300
  br label %305

304:                                              ; preds = %300
  br label %305

305:                                              ; preds = %304, %303
  %306 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %304 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %303 ]
  %307 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %301, 1
  %308 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %306, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %307, ptr %308, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %309 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %310 = add i32 %309, 1
  %311 = icmp sge i32 %310, 2
  %312 = add i32 %309, -1
  %313 = select i1 %311, i32 %312, i32 %310
  store i32 %313, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %314 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %315 = add i32 %314, 1
  %316 = icmp sge i32 %315, 2
  %317 = add i32 %314, -1
  %318 = select i1 %316, i32 %317, i32 %315
  store i32 %318, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %319 = add i64 %294, 1
  br label %293

320:                                              ; preds = %293
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %321 = load i32, ptr @_anonymous0, align 4
  %322 = add i32 %321, 1
  %323 = icmp sge i32 %322, 1
  %324 = select i1 %323, i32 %321, i32 %322
  store i32 %324, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %325

325:                                              ; preds = %337, %320
  %326 = phi i64 [ %351, %337 ], [ 0, %320 ]
  %327 = icmp slt i64 %326, 16
  br i1 %327, label %328, label %352

328:                                              ; preds = %325
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %329 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %329, label %330 [
    i32 0, label %330
    i32 1, label %331
  ]

330:                                              ; preds = %328, %328
  br label %332

331:                                              ; preds = %328
  br label %332

332:                                              ; preds = %331, %330
  %333 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %331 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %330 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %334 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %334, label %335 [
    i32 0, label %335
    i32 1, label %336
  ]

335:                                              ; preds = %332, %332
  br label %337

336:                                              ; preds = %332
  br label %337

337:                                              ; preds = %336, %335
  %338 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %336 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %335 ]
  %339 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %333, 1
  %340 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %338, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %339, ptr %340, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %341 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %342 = add i32 %341, 1
  %343 = icmp sge i32 %342, 2
  %344 = add i32 %341, -1
  %345 = select i1 %343, i32 %344, i32 %342
  store i32 %345, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %346 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %347 = add i32 %346, 1
  %348 = icmp sge i32 %347, 2
  %349 = add i32 %346, -1
  %350 = select i1 %348, i32 %349, i32 %347
  store i32 %350, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %351 = add i64 %326, 1
  br label %325

352:                                              ; preds = %325
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %353 = load i32, ptr @_anonymous0, align 4
  %354 = add i32 %353, 1
  %355 = icmp sge i32 %354, 1
  %356 = select i1 %355, i32 %353, i32 %354
  store i32 %356, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %357

357:                                              ; preds = %369, %352
  %358 = phi i64 [ %383, %369 ], [ 0, %352 ]
  %359 = icmp slt i64 %358, 16
  br i1 %359, label %360, label %384

360:                                              ; preds = %357
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %361 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %361, label %362 [
    i32 0, label %362
    i32 1, label %363
  ]

362:                                              ; preds = %360, %360
  br label %364

363:                                              ; preds = %360
  br label %364

364:                                              ; preds = %363, %362
  %365 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %363 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %362 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %366 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %366, label %367 [
    i32 0, label %367
    i32 1, label %368
  ]

367:                                              ; preds = %364, %364
  br label %369

368:                                              ; preds = %364
  br label %369

369:                                              ; preds = %368, %367
  %370 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %368 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %367 ]
  %371 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %365, 1
  %372 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %370, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %371, ptr %372, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %373 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %374 = add i32 %373, 1
  %375 = icmp sge i32 %374, 2
  %376 = add i32 %373, -1
  %377 = select i1 %375, i32 %376, i32 %374
  store i32 %377, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %378 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %379 = add i32 %378, 1
  %380 = icmp sge i32 %379, 2
  %381 = add i32 %378, -1
  %382 = select i1 %380, i32 %381, i32 %379
  store i32 %382, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %383 = add i64 %358, 1
  br label %357

384:                                              ; preds = %357
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %385 = load i32, ptr @_anonymous0, align 4
  %386 = add i32 %385, 1
  %387 = icmp sge i32 %386, 1
  %388 = select i1 %387, i32 %385, i32 %386
  store i32 %388, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %389

389:                                              ; preds = %401, %384
  %390 = phi i64 [ %415, %401 ], [ 0, %384 ]
  %391 = icmp slt i64 %390, 16
  br i1 %391, label %392, label %416

392:                                              ; preds = %389
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %393 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %393, label %394 [
    i32 0, label %394
    i32 1, label %395
  ]

394:                                              ; preds = %392, %392
  br label %396

395:                                              ; preds = %392
  br label %396

396:                                              ; preds = %395, %394
  %397 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %395 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %394 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %398 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %398, label %399 [
    i32 0, label %399
    i32 1, label %400
  ]

399:                                              ; preds = %396, %396
  br label %401

400:                                              ; preds = %396
  br label %401

401:                                              ; preds = %400, %399
  %402 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %400 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %399 ]
  %403 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %397, 1
  %404 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %402, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %403, ptr %404, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %405 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %406 = add i32 %405, 1
  %407 = icmp sge i32 %406, 2
  %408 = add i32 %405, -1
  %409 = select i1 %407, i32 %408, i32 %406
  store i32 %409, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %410 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %411 = add i32 %410, 1
  %412 = icmp sge i32 %411, 2
  %413 = add i32 %410, -1
  %414 = select i1 %412, i32 %413, i32 %411
  store i32 %414, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %415 = add i64 %390, 1
  br label %389

416:                                              ; preds = %389
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %417 = load i32, ptr @_anonymous0, align 4
  %418 = add i32 %417, 1
  %419 = icmp sge i32 %418, 1
  %420 = select i1 %419, i32 %417, i32 %418
  store i32 %420, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %421

421:                                              ; preds = %433, %416
  %422 = phi i64 [ %447, %433 ], [ 0, %416 ]
  %423 = icmp slt i64 %422, 16
  br i1 %423, label %424, label %448

424:                                              ; preds = %421
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %425 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %425, label %426 [
    i32 0, label %426
    i32 1, label %427
  ]

426:                                              ; preds = %424, %424
  br label %428

427:                                              ; preds = %424
  br label %428

428:                                              ; preds = %427, %426
  %429 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %427 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %426 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %430 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %430, label %431 [
    i32 0, label %431
    i32 1, label %432
  ]

431:                                              ; preds = %428, %428
  br label %433

432:                                              ; preds = %428
  br label %433

433:                                              ; preds = %432, %431
  %434 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %432 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %431 ]
  %435 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %429, 1
  %436 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %434, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %435, ptr %436, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %437 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %438 = add i32 %437, 1
  %439 = icmp sge i32 %438, 2
  %440 = add i32 %437, -1
  %441 = select i1 %439, i32 %440, i32 %438
  store i32 %441, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %442 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %443 = add i32 %442, 1
  %444 = icmp sge i32 %443, 2
  %445 = add i32 %442, -1
  %446 = select i1 %444, i32 %445, i32 %443
  store i32 %446, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %447 = add i64 %422, 1
  br label %421

448:                                              ; preds = %421
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %449 = load i32, ptr @_anonymous0, align 4
  %450 = add i32 %449, 1
  %451 = icmp sge i32 %450, 1
  %452 = select i1 %451, i32 %449, i32 %450
  store i32 %452, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %453

453:                                              ; preds = %465, %448
  %454 = phi i64 [ %479, %465 ], [ 0, %448 ]
  %455 = icmp slt i64 %454, 16
  br i1 %455, label %456, label %480

456:                                              ; preds = %453
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %457 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %457, label %458 [
    i32 0, label %458
    i32 1, label %459
  ]

458:                                              ; preds = %456, %456
  br label %460

459:                                              ; preds = %456
  br label %460

460:                                              ; preds = %459, %458
  %461 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %459 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %458 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %462 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %462, label %463 [
    i32 0, label %463
    i32 1, label %464
  ]

463:                                              ; preds = %460, %460
  br label %465

464:                                              ; preds = %460
  br label %465

465:                                              ; preds = %464, %463
  %466 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %464 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %463 ]
  %467 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %461, 1
  %468 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %466, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %467, ptr %468, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %469 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %470 = add i32 %469, 1
  %471 = icmp sge i32 %470, 2
  %472 = add i32 %469, -1
  %473 = select i1 %471, i32 %472, i32 %470
  store i32 %473, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %474 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %475 = add i32 %474, 1
  %476 = icmp sge i32 %475, 2
  %477 = add i32 %474, -1
  %478 = select i1 %476, i32 %477, i32 %475
  store i32 %478, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %479 = add i64 %454, 1
  br label %453

480:                                              ; preds = %453
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %481 = load i32, ptr @_anonymous0, align 4
  %482 = add i32 %481, 1
  %483 = icmp sge i32 %482, 1
  %484 = select i1 %483, i32 %481, i32 %482
  store i32 %484, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %485

485:                                              ; preds = %497, %480
  %486 = phi i64 [ %511, %497 ], [ 0, %480 ]
  %487 = icmp slt i64 %486, 16
  br i1 %487, label %488, label %512

488:                                              ; preds = %485
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %489 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %489, label %490 [
    i32 0, label %490
    i32 1, label %491
  ]

490:                                              ; preds = %488, %488
  br label %492

491:                                              ; preds = %488
  br label %492

492:                                              ; preds = %491, %490
  %493 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %491 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %490 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %494 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %494, label %495 [
    i32 0, label %495
    i32 1, label %496
  ]

495:                                              ; preds = %492, %492
  br label %497

496:                                              ; preds = %492
  br label %497

497:                                              ; preds = %496, %495
  %498 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %496 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %495 ]
  %499 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %493, 1
  %500 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %498, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %499, ptr %500, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %501 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %502 = add i32 %501, 1
  %503 = icmp sge i32 %502, 2
  %504 = add i32 %501, -1
  %505 = select i1 %503, i32 %504, i32 %502
  store i32 %505, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %506 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %507 = add i32 %506, 1
  %508 = icmp sge i32 %507, 2
  %509 = add i32 %506, -1
  %510 = select i1 %508, i32 %509, i32 %507
  store i32 %510, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %511 = add i64 %486, 1
  br label %485

512:                                              ; preds = %485
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %513 = load i32, ptr @_anonymous0, align 4
  %514 = add i32 %513, 1
  %515 = icmp sge i32 %514, 1
  %516 = select i1 %515, i32 %513, i32 %514
  store i32 %516, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %517

517:                                              ; preds = %529, %512
  %518 = phi i64 [ %543, %529 ], [ 0, %512 ]
  %519 = icmp slt i64 %518, 16
  br i1 %519, label %520, label %544

520:                                              ; preds = %517
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %521 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %521, label %522 [
    i32 0, label %522
    i32 1, label %523
  ]

522:                                              ; preds = %520, %520
  br label %524

523:                                              ; preds = %520
  br label %524

524:                                              ; preds = %523, %522
  %525 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %523 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %522 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %526 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %526, label %527 [
    i32 0, label %527
    i32 1, label %528
  ]

527:                                              ; preds = %524, %524
  br label %529

528:                                              ; preds = %524
  br label %529

529:                                              ; preds = %528, %527
  %530 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %528 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %527 ]
  %531 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %525, 1
  %532 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %530, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %531, ptr %532, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %533 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %534 = add i32 %533, 1
  %535 = icmp sge i32 %534, 2
  %536 = add i32 %533, -1
  %537 = select i1 %535, i32 %536, i32 %534
  store i32 %537, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %538 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %539 = add i32 %538, 1
  %540 = icmp sge i32 %539, 2
  %541 = add i32 %538, -1
  %542 = select i1 %540, i32 %541, i32 %539
  store i32 %542, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %543 = add i64 %518, 1
  br label %517

544:                                              ; preds = %517
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %545 = load i32, ptr @_anonymous0, align 4
  %546 = add i32 %545, 1
  %547 = icmp sge i32 %546, 1
  %548 = select i1 %547, i32 %545, i32 %546
  store i32 %548, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %549

549:                                              ; preds = %561, %544
  %550 = phi i64 [ %575, %561 ], [ 0, %544 ]
  %551 = icmp slt i64 %550, 16
  br i1 %551, label %552, label %576

552:                                              ; preds = %549
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %553 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %553, label %554 [
    i32 0, label %554
    i32 1, label %555
  ]

554:                                              ; preds = %552, %552
  br label %556

555:                                              ; preds = %552
  br label %556

556:                                              ; preds = %555, %554
  %557 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %555 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %554 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %558 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %558, label %559 [
    i32 0, label %559
    i32 1, label %560
  ]

559:                                              ; preds = %556, %556
  br label %561

560:                                              ; preds = %556
  br label %561

561:                                              ; preds = %560, %559
  %562 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %560 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %559 ]
  %563 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %557, 1
  %564 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %562, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %563, ptr %564, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %565 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %566 = add i32 %565, 1
  %567 = icmp sge i32 %566, 2
  %568 = add i32 %565, -1
  %569 = select i1 %567, i32 %568, i32 %566
  store i32 %569, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %570 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %571 = add i32 %570, 1
  %572 = icmp sge i32 %571, 2
  %573 = add i32 %570, -1
  %574 = select i1 %572, i32 %573, i32 %571
  store i32 %574, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %575 = add i64 %550, 1
  br label %549

576:                                              ; preds = %549
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %577 = load i32, ptr @_anonymous0, align 4
  %578 = add i32 %577, 1
  %579 = icmp sge i32 %578, 1
  %580 = select i1 %579, i32 %577, i32 %578
  store i32 %580, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %581

581:                                              ; preds = %593, %576
  %582 = phi i64 [ %607, %593 ], [ 0, %576 ]
  %583 = icmp slt i64 %582, 16
  br i1 %583, label %584, label %608

584:                                              ; preds = %581
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %585 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %585, label %586 [
    i32 0, label %586
    i32 1, label %587
  ]

586:                                              ; preds = %584, %584
  br label %588

587:                                              ; preds = %584
  br label %588

588:                                              ; preds = %587, %586
  %589 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %587 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %586 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %590 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %590, label %591 [
    i32 0, label %591
    i32 1, label %592
  ]

591:                                              ; preds = %588, %588
  br label %593

592:                                              ; preds = %588
  br label %593

593:                                              ; preds = %592, %591
  %594 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %592 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %591 ]
  %595 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %589, 1
  %596 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %594, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %595, ptr %596, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %597 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %598 = add i32 %597, 1
  %599 = icmp sge i32 %598, 2
  %600 = add i32 %597, -1
  %601 = select i1 %599, i32 %600, i32 %598
  store i32 %601, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %602 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %603 = add i32 %602, 1
  %604 = icmp sge i32 %603, 2
  %605 = add i32 %602, -1
  %606 = select i1 %604, i32 %605, i32 %603
  store i32 %606, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %607 = add i64 %582, 1
  br label %581

608:                                              ; preds = %581
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %609 = load i32, ptr @_anonymous0, align 4
  %610 = add i32 %609, 1
  %611 = icmp sge i32 %610, 1
  %612 = select i1 %611, i32 %609, i32 %610
  store i32 %612, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %613

613:                                              ; preds = %625, %608
  %614 = phi i64 [ %639, %625 ], [ 0, %608 ]
  %615 = icmp slt i64 %614, 16
  br i1 %615, label %616, label %640

616:                                              ; preds = %613
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %617 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %617, label %618 [
    i32 0, label %618
    i32 1, label %619
  ]

618:                                              ; preds = %616, %616
  br label %620

619:                                              ; preds = %616
  br label %620

620:                                              ; preds = %619, %618
  %621 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %619 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %618 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %622 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %622, label %623 [
    i32 0, label %623
    i32 1, label %624
  ]

623:                                              ; preds = %620, %620
  br label %625

624:                                              ; preds = %620
  br label %625

625:                                              ; preds = %624, %623
  %626 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %624 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %623 ]
  %627 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %621, 1
  %628 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %626, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %627, ptr %628, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %629 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %630 = add i32 %629, 1
  %631 = icmp sge i32 %630, 2
  %632 = add i32 %629, -1
  %633 = select i1 %631, i32 %632, i32 %630
  store i32 %633, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %634 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %635 = add i32 %634, 1
  %636 = icmp sge i32 %635, 2
  %637 = add i32 %634, -1
  %638 = select i1 %636, i32 %637, i32 %635
  store i32 %638, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %639 = add i64 %614, 1
  br label %613

640:                                              ; preds = %613
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %641 = load i32, ptr @_anonymous0, align 4
  %642 = add i32 %641, 1
  %643 = icmp sge i32 %642, 1
  %644 = select i1 %643, i32 %641, i32 %642
  store i32 %644, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %645

645:                                              ; preds = %657, %640
  %646 = phi i64 [ %671, %657 ], [ 0, %640 ]
  %647 = icmp slt i64 %646, 16
  br i1 %647, label %648, label %672

648:                                              ; preds = %645
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %649 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %649, label %650 [
    i32 0, label %650
    i32 1, label %651
  ]

650:                                              ; preds = %648, %648
  br label %652

651:                                              ; preds = %648
  br label %652

652:                                              ; preds = %651, %650
  %653 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %651 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %650 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %654 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %654, label %655 [
    i32 0, label %655
    i32 1, label %656
  ]

655:                                              ; preds = %652, %652
  br label %657

656:                                              ; preds = %652
  br label %657

657:                                              ; preds = %656, %655
  %658 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %656 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %655 ]
  %659 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %653, 1
  %660 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %658, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %659, ptr %660, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %661 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %662 = add i32 %661, 1
  %663 = icmp sge i32 %662, 2
  %664 = add i32 %661, -1
  %665 = select i1 %663, i32 %664, i32 %662
  store i32 %665, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %666 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %667 = add i32 %666, 1
  %668 = icmp sge i32 %667, 2
  %669 = add i32 %666, -1
  %670 = select i1 %668, i32 %669, i32 %667
  store i32 %670, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %671 = add i64 %646, 1
  br label %645

672:                                              ; preds = %645
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %673 = load i32, ptr @_anonymous0, align 4
  %674 = add i32 %673, 1
  %675 = icmp sge i32 %674, 1
  %676 = select i1 %675, i32 %673, i32 %674
  store i32 %676, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %677

677:                                              ; preds = %689, %672
  %678 = phi i64 [ %703, %689 ], [ 0, %672 ]
  %679 = icmp slt i64 %678, 16
  br i1 %679, label %680, label %704

680:                                              ; preds = %677
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %681 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %681, label %682 [
    i32 0, label %682
    i32 1, label %683
  ]

682:                                              ; preds = %680, %680
  br label %684

683:                                              ; preds = %680
  br label %684

684:                                              ; preds = %683, %682
  %685 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %683 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %682 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %686 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %686, label %687 [
    i32 0, label %687
    i32 1, label %688
  ]

687:                                              ; preds = %684, %684
  br label %689

688:                                              ; preds = %684
  br label %689

689:                                              ; preds = %688, %687
  %690 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %688 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %687 ]
  %691 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %685, 1
  %692 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %690, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %691, ptr %692, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %693 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %694 = add i32 %693, 1
  %695 = icmp sge i32 %694, 2
  %696 = add i32 %693, -1
  %697 = select i1 %695, i32 %696, i32 %694
  store i32 %697, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %698 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %699 = add i32 %698, 1
  %700 = icmp sge i32 %699, 2
  %701 = add i32 %698, -1
  %702 = select i1 %700, i32 %701, i32 %699
  store i32 %702, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %703 = add i64 %678, 1
  br label %677

704:                                              ; preds = %677
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %705 = load i32, ptr @_anonymous0, align 4
  %706 = add i32 %705, 1
  %707 = icmp sge i32 %706, 1
  %708 = select i1 %707, i32 %705, i32 %706
  store i32 %708, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %709

709:                                              ; preds = %721, %704
  %710 = phi i64 [ %735, %721 ], [ 0, %704 ]
  %711 = icmp slt i64 %710, 16
  br i1 %711, label %712, label %736

712:                                              ; preds = %709
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %713 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %713, label %714 [
    i32 0, label %714
    i32 1, label %715
  ]

714:                                              ; preds = %712, %712
  br label %716

715:                                              ; preds = %712
  br label %716

716:                                              ; preds = %715, %714
  %717 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %715 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %714 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %718 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %718, label %719 [
    i32 0, label %719
    i32 1, label %720
  ]

719:                                              ; preds = %716, %716
  br label %721

720:                                              ; preds = %716
  br label %721

721:                                              ; preds = %720, %719
  %722 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %720 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %719 ]
  %723 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %717, 1
  %724 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %722, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %723, ptr %724, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %725 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %726 = add i32 %725, 1
  %727 = icmp sge i32 %726, 2
  %728 = add i32 %725, -1
  %729 = select i1 %727, i32 %728, i32 %726
  store i32 %729, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %730 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %731 = add i32 %730, 1
  %732 = icmp sge i32 %731, 2
  %733 = add i32 %730, -1
  %734 = select i1 %732, i32 %733, i32 %731
  store i32 %734, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %735 = add i64 %710, 1
  br label %709

736:                                              ; preds = %709
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %737 = load i32, ptr @_anonymous0, align 4
  %738 = add i32 %737, 1
  %739 = icmp sge i32 %738, 1
  %740 = select i1 %739, i32 %737, i32 %738
  store i32 %740, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %741

741:                                              ; preds = %753, %736
  %742 = phi i64 [ %767, %753 ], [ 0, %736 ]
  %743 = icmp slt i64 %742, 16
  br i1 %743, label %744, label %768

744:                                              ; preds = %741
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %745 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %745, label %746 [
    i32 0, label %746
    i32 1, label %747
  ]

746:                                              ; preds = %744, %744
  br label %748

747:                                              ; preds = %744
  br label %748

748:                                              ; preds = %747, %746
  %749 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %747 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %746 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %750 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %750, label %751 [
    i32 0, label %751
    i32 1, label %752
  ]

751:                                              ; preds = %748, %748
  br label %753

752:                                              ; preds = %748
  br label %753

753:                                              ; preds = %752, %751
  %754 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %752 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %751 ]
  %755 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %749, 1
  %756 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %754, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %755, ptr %756, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %757 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %758 = add i32 %757, 1
  %759 = icmp sge i32 %758, 2
  %760 = add i32 %757, -1
  %761 = select i1 %759, i32 %760, i32 %758
  store i32 %761, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %762 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %763 = add i32 %762, 1
  %764 = icmp sge i32 %763, 2
  %765 = add i32 %762, -1
  %766 = select i1 %764, i32 %765, i32 %763
  store i32 %766, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %767 = add i64 %742, 1
  br label %741

768:                                              ; preds = %741
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %769 = load i32, ptr @_anonymous0, align 4
  %770 = add i32 %769, 1
  %771 = icmp sge i32 %770, 1
  %772 = select i1 %771, i32 %769, i32 %770
  store i32 %772, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %773

773:                                              ; preds = %785, %768
  %774 = phi i64 [ %799, %785 ], [ 0, %768 ]
  %775 = icmp slt i64 %774, 16
  br i1 %775, label %776, label %800

776:                                              ; preds = %773
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %777 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %777, label %778 [
    i32 0, label %778
    i32 1, label %779
  ]

778:                                              ; preds = %776, %776
  br label %780

779:                                              ; preds = %776
  br label %780

780:                                              ; preds = %779, %778
  %781 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %779 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %778 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %782 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %782, label %783 [
    i32 0, label %783
    i32 1, label %784
  ]

783:                                              ; preds = %780, %780
  br label %785

784:                                              ; preds = %780
  br label %785

785:                                              ; preds = %784, %783
  %786 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %784 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %783 ]
  %787 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %781, 1
  %788 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %786, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %787, ptr %788, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %789 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %790 = add i32 %789, 1
  %791 = icmp sge i32 %790, 2
  %792 = add i32 %789, -1
  %793 = select i1 %791, i32 %792, i32 %790
  store i32 %793, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %794 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %795 = add i32 %794, 1
  %796 = icmp sge i32 %795, 2
  %797 = add i32 %794, -1
  %798 = select i1 %796, i32 %797, i32 %795
  store i32 %798, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %799 = add i64 %774, 1
  br label %773

800:                                              ; preds = %773
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %801 = load i32, ptr @_anonymous0, align 4
  %802 = add i32 %801, 1
  %803 = icmp sge i32 %802, 1
  %804 = select i1 %803, i32 %801, i32 %802
  store i32 %804, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %805

805:                                              ; preds = %817, %800
  %806 = phi i64 [ %831, %817 ], [ 0, %800 ]
  %807 = icmp slt i64 %806, 16
  br i1 %807, label %808, label %832

808:                                              ; preds = %805
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %809 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %809, label %810 [
    i32 0, label %810
    i32 1, label %811
  ]

810:                                              ; preds = %808, %808
  br label %812

811:                                              ; preds = %808
  br label %812

812:                                              ; preds = %811, %810
  %813 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %811 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %810 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %814 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %814, label %815 [
    i32 0, label %815
    i32 1, label %816
  ]

815:                                              ; preds = %812, %812
  br label %817

816:                                              ; preds = %812
  br label %817

817:                                              ; preds = %816, %815
  %818 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %816 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %815 ]
  %819 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %813, 1
  %820 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %818, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %819, ptr %820, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %821 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %822 = add i32 %821, 1
  %823 = icmp sge i32 %822, 2
  %824 = add i32 %821, -1
  %825 = select i1 %823, i32 %824, i32 %822
  store i32 %825, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %826 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %827 = add i32 %826, 1
  %828 = icmp sge i32 %827, 2
  %829 = add i32 %826, -1
  %830 = select i1 %828, i32 %829, i32 %827
  store i32 %830, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %831 = add i64 %806, 1
  br label %805

832:                                              ; preds = %805
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %833 = load i32, ptr @_anonymous0, align 4
  %834 = add i32 %833, 1
  %835 = icmp sge i32 %834, 1
  %836 = select i1 %835, i32 %833, i32 %834
  store i32 %836, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %837

837:                                              ; preds = %849, %832
  %838 = phi i64 [ %863, %849 ], [ 0, %832 ]
  %839 = icmp slt i64 %838, 16
  br i1 %839, label %840, label %864

840:                                              ; preds = %837
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %841 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %841, label %842 [
    i32 0, label %842
    i32 1, label %843
  ]

842:                                              ; preds = %840, %840
  br label %844

843:                                              ; preds = %840
  br label %844

844:                                              ; preds = %843, %842
  %845 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %843 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %842 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %846 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %846, label %847 [
    i32 0, label %847
    i32 1, label %848
  ]

847:                                              ; preds = %844, %844
  br label %849

848:                                              ; preds = %844
  br label %849

849:                                              ; preds = %848, %847
  %850 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %848 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %847 ]
  %851 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %845, 1
  %852 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %850, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %851, ptr %852, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %853 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %854 = add i32 %853, 1
  %855 = icmp sge i32 %854, 2
  %856 = add i32 %853, -1
  %857 = select i1 %855, i32 %856, i32 %854
  store i32 %857, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %858 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %859 = add i32 %858, 1
  %860 = icmp sge i32 %859, 2
  %861 = add i32 %858, -1
  %862 = select i1 %860, i32 %861, i32 %859
  store i32 %862, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %863 = add i64 %838, 1
  br label %837

864:                                              ; preds = %837
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %865 = load i32, ptr @_anonymous0, align 4
  %866 = add i32 %865, 1
  %867 = icmp sge i32 %866, 1
  %868 = select i1 %867, i32 %865, i32 %866
  store i32 %868, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %869

869:                                              ; preds = %881, %864
  %870 = phi i64 [ %895, %881 ], [ 0, %864 ]
  %871 = icmp slt i64 %870, 16
  br i1 %871, label %872, label %896

872:                                              ; preds = %869
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %873 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %873, label %874 [
    i32 0, label %874
    i32 1, label %875
  ]

874:                                              ; preds = %872, %872
  br label %876

875:                                              ; preds = %872
  br label %876

876:                                              ; preds = %875, %874
  %877 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %875 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %874 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %878 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %878, label %879 [
    i32 0, label %879
    i32 1, label %880
  ]

879:                                              ; preds = %876, %876
  br label %881

880:                                              ; preds = %876
  br label %881

881:                                              ; preds = %880, %879
  %882 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %880 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %879 ]
  %883 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %877, 1
  %884 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %882, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %883, ptr %884, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %885 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %886 = add i32 %885, 1
  %887 = icmp sge i32 %886, 2
  %888 = add i32 %885, -1
  %889 = select i1 %887, i32 %888, i32 %886
  store i32 %889, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %890 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %891 = add i32 %890, 1
  %892 = icmp sge i32 %891, 2
  %893 = add i32 %890, -1
  %894 = select i1 %892, i32 %893, i32 %891
  store i32 %894, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %895 = add i64 %870, 1
  br label %869

896:                                              ; preds = %869
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %897 = load i32, ptr @_anonymous0, align 4
  %898 = add i32 %897, 1
  %899 = icmp sge i32 %898, 1
  %900 = select i1 %899, i32 %897, i32 %898
  store i32 %900, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %901

901:                                              ; preds = %913, %896
  %902 = phi i64 [ %927, %913 ], [ 0, %896 ]
  %903 = icmp slt i64 %902, 16
  br i1 %903, label %904, label %928

904:                                              ; preds = %901
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %905 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %905, label %906 [
    i32 0, label %906
    i32 1, label %907
  ]

906:                                              ; preds = %904, %904
  br label %908

907:                                              ; preds = %904
  br label %908

908:                                              ; preds = %907, %906
  %909 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %907 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %906 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %910 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %910, label %911 [
    i32 0, label %911
    i32 1, label %912
  ]

911:                                              ; preds = %908, %908
  br label %913

912:                                              ; preds = %908
  br label %913

913:                                              ; preds = %912, %911
  %914 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %912 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %911 ]
  %915 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %909, 1
  %916 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %914, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %915, ptr %916, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %917 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %918 = add i32 %917, 1
  %919 = icmp sge i32 %918, 2
  %920 = add i32 %917, -1
  %921 = select i1 %919, i32 %920, i32 %918
  store i32 %921, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %922 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %923 = add i32 %922, 1
  %924 = icmp sge i32 %923, 2
  %925 = add i32 %922, -1
  %926 = select i1 %924, i32 %925, i32 %923
  store i32 %926, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %927 = add i64 %902, 1
  br label %901

928:                                              ; preds = %901
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %929 = load i32, ptr @_anonymous0, align 4
  %930 = add i32 %929, 1
  %931 = icmp sge i32 %930, 1
  %932 = select i1 %931, i32 %929, i32 %930
  store i32 %932, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %933

933:                                              ; preds = %945, %928
  %934 = phi i64 [ %959, %945 ], [ 0, %928 ]
  %935 = icmp slt i64 %934, 16
  br i1 %935, label %936, label %960

936:                                              ; preds = %933
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %937 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %937, label %938 [
    i32 0, label %938
    i32 1, label %939
  ]

938:                                              ; preds = %936, %936
  br label %940

939:                                              ; preds = %936
  br label %940

940:                                              ; preds = %939, %938
  %941 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %939 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %938 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %942 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %942, label %943 [
    i32 0, label %943
    i32 1, label %944
  ]

943:                                              ; preds = %940, %940
  br label %945

944:                                              ; preds = %940
  br label %945

945:                                              ; preds = %944, %943
  %946 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %944 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %943 ]
  %947 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %941, 1
  %948 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %946, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %947, ptr %948, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %949 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %950 = add i32 %949, 1
  %951 = icmp sge i32 %950, 2
  %952 = add i32 %949, -1
  %953 = select i1 %951, i32 %952, i32 %950
  store i32 %953, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %954 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %955 = add i32 %954, 1
  %956 = icmp sge i32 %955, 2
  %957 = add i32 %954, -1
  %958 = select i1 %956, i32 %957, i32 %955
  store i32 %958, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %959 = add i64 %934, 1
  br label %933

960:                                              ; preds = %933
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %961 = load i32, ptr @_anonymous0, align 4
  %962 = add i32 %961, 1
  %963 = icmp sge i32 %962, 1
  %964 = select i1 %963, i32 %961, i32 %962
  store i32 %964, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %965

965:                                              ; preds = %977, %960
  %966 = phi i64 [ %991, %977 ], [ 0, %960 ]
  %967 = icmp slt i64 %966, 16
  br i1 %967, label %968, label %992

968:                                              ; preds = %965
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %969 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %969, label %970 [
    i32 0, label %970
    i32 1, label %971
  ]

970:                                              ; preds = %968, %968
  br label %972

971:                                              ; preds = %968
  br label %972

972:                                              ; preds = %971, %970
  %973 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %971 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %970 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %974 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %974, label %975 [
    i32 0, label %975
    i32 1, label %976
  ]

975:                                              ; preds = %972, %972
  br label %977

976:                                              ; preds = %972
  br label %977

977:                                              ; preds = %976, %975
  %978 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %976 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %975 ]
  %979 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %973, 1
  %980 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %978, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %979, ptr %980, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %981 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %982 = add i32 %981, 1
  %983 = icmp sge i32 %982, 2
  %984 = add i32 %981, -1
  %985 = select i1 %983, i32 %984, i32 %982
  store i32 %985, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %986 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %987 = add i32 %986, 1
  %988 = icmp sge i32 %987, 2
  %989 = add i32 %986, -1
  %990 = select i1 %988, i32 %989, i32 %987
  store i32 %990, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %991 = add i64 %966, 1
  br label %965

992:                                              ; preds = %965
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %993 = load i32, ptr @_anonymous0, align 4
  %994 = add i32 %993, 1
  %995 = icmp sge i32 %994, 1
  %996 = select i1 %995, i32 %993, i32 %994
  store i32 %996, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_buff_0)
  br label %997

997:                                              ; preds = %1009, %992
  %998 = phi i64 [ %1023, %1009 ], [ 0, %992 ]
  %999 = icmp slt i64 %998, 16
  br i1 %999, label %1000, label %1024

1000:                                             ; preds = %997
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %1001 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %1001, label %1002 [
    i32 0, label %1002
    i32 1, label %1003
  ]

1002:                                             ; preds = %1000, %1000
  br label %1004

1003:                                             ; preds = %1000
  br label %1004

1004:                                             ; preds = %1003, %1002
  %1005 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %1003 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %1002 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %1006 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %1006, label %1007 [
    i32 0, label %1007
    i32 1, label %1008
  ]

1007:                                             ; preds = %1004, %1004
  br label %1009

1008:                                             ; preds = %1004
  br label %1009

1009:                                             ; preds = %1008, %1007
  %1010 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %1008 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %1007 ]
  %1011 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1005, 1
  %1012 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1010, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %1011, ptr %1012, ptr @C_L1L2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %1013 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %1014 = add i32 %1013, 1
  %1015 = icmp sge i32 %1014, 2
  %1016 = add i32 %1013, -1
  %1017 = select i1 %1015, i32 %1016, i32 %1014
  store i32 %1017, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %1018 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %1019 = add i32 %1018, 1
  %1020 = icmp sge i32 %1019, 2
  %1021 = add i32 %1018, -1
  %1022 = select i1 %1020, i32 %1021, i32 %1019
  store i32 %1022, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %1023 = add i64 %998, 1
  br label %997

1024:                                             ; preds = %997
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %1025 = load i32, ptr @_anonymous0, align 4
  %1026 = add i32 %1025, 1
  %1027 = icmp sge i32 %1026, 1
  %1028 = select i1 %1027, i32 %1025, i32 %1026
  store i32 %1028, ptr @_anonymous0, align 4
  %1029 = add i64 %2, 1
  br label %1

1030:                                             ; preds = %1
  ret void
}

declare void @llvm.aie2p.acquire(i32, i32) addrspace(0)

declare void @zero_scalar_i16(ptr) addrspace(0)

declare void @matmul_scalar_i8_i16(ptr, ptr, ptr) addrspace(0)

declare void @llvm.aie2p.release(i32, i32) addrspace(0)

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
