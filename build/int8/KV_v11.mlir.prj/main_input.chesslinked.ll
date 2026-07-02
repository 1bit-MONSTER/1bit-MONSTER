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

1:                                                ; preds = %148, %0
  %2 = phi i64 [ %153, %148 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %154

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_7_0_buff_0)
  br label %5

5:                                                ; preds = %35, %4
  %6 = phi i64 [ %36, %35 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 2
  br i1 %7, label %8, label %37

8:                                                ; preds = %20, %5
  %9 = phi i64 [ %34, %20 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 8
  br i1 %10, label %11, label %35

11:                                               ; preds = %8
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %12 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  switch i32 %12, label %13 [
    i32 0, label %13
    i32 1, label %14
  ]

13:                                               ; preds = %11, %11
  br label %15

14:                                               ; preds = %11
  br label %15

15:                                               ; preds = %14, %13
  %16 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %14 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %13 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %17 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %17, label %18 [
    i32 0, label %18
    i32 1, label %19
  ]

18:                                               ; preds = %15, %15
  br label %20

19:                                               ; preds = %15
  br label %20

20:                                               ; preds = %19, %18
  %21 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %18 ]
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %22, ptr %23, ptr @C_L1L2_7_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %24 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %25 = add i32 %24, 1
  %26 = icmp sge i32 %25, 2
  %27 = add i32 %24, -1
  %28 = select i1 %26, i32 %27, i32 %25
  store i32 %28, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %29 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %30 = add i32 %29, 1
  %31 = icmp sge i32 %30, 2
  %32 = add i32 %29, -1
  %33 = select i1 %31, i32 %32, i32 %30
  store i32 %33, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %34 = add i64 %9, 1
  br label %8

35:                                               ; preds = %8
  %36 = add i64 %6, 1
  br label %5

37:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %38 = load i32, ptr @_anonymous7, align 4
  %39 = add i32 %38, 1
  %40 = icmp sge i32 %39, 1
  %41 = select i1 %40, i32 %38, i32 %39
  store i32 %41, ptr @_anonymous7, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_7_0_buff_0)
  br label %42

42:                                               ; preds = %72, %37
  %43 = phi i64 [ %73, %72 ], [ 0, %37 ]
  %44 = icmp slt i64 %43, 2
  br i1 %44, label %45, label %74

45:                                               ; preds = %57, %42
  %46 = phi i64 [ %71, %57 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 8
  br i1 %47, label %48, label %72

48:                                               ; preds = %45
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  switch i32 %49, label %50 [
    i32 0, label %50
    i32 1, label %51
  ]

50:                                               ; preds = %48, %48
  br label %52

51:                                               ; preds = %48
  br label %52

52:                                               ; preds = %51, %50
  %53 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %51 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %50 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %54, label %55 [
    i32 0, label %55
    i32 1, label %56
  ]

55:                                               ; preds = %52, %52
  br label %57

56:                                               ; preds = %52
  br label %57

57:                                               ; preds = %56, %55
  %58 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %56 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %55 ]
  %59 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %59, ptr %60, ptr @C_L1L2_7_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %61 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %62 = add i32 %61, 1
  %63 = icmp sge i32 %62, 2
  %64 = add i32 %61, -1
  %65 = select i1 %63, i32 %64, i32 %62
  store i32 %65, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %66 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 2
  %69 = add i32 %66, -1
  %70 = select i1 %68, i32 %69, i32 %67
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %71 = add i64 %46, 1
  br label %45

72:                                               ; preds = %45
  %73 = add i64 %43, 1
  br label %42

74:                                               ; preds = %42
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %75 = load i32, ptr @_anonymous7, align 4
  %76 = add i32 %75, 1
  %77 = icmp sge i32 %76, 1
  %78 = select i1 %77, i32 %75, i32 %76
  store i32 %78, ptr @_anonymous7, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_7_0_buff_0)
  br label %79

79:                                               ; preds = %109, %74
  %80 = phi i64 [ %110, %109 ], [ 0, %74 ]
  %81 = icmp slt i64 %80, 2
  br i1 %81, label %82, label %111

82:                                               ; preds = %94, %79
  %83 = phi i64 [ %108, %94 ], [ 0, %79 ]
  %84 = icmp slt i64 %83, 8
  br i1 %84, label %85, label %109

85:                                               ; preds = %82
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %86 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  switch i32 %86, label %87 [
    i32 0, label %87
    i32 1, label %88
  ]

87:                                               ; preds = %85, %85
  br label %89

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %88, %87
  %90 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %88 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %87 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %91 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %91, label %92 [
    i32 0, label %92
    i32 1, label %93
  ]

92:                                               ; preds = %89, %89
  br label %94

93:                                               ; preds = %89
  br label %94

94:                                               ; preds = %93, %92
  %95 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %93 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %92 ]
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, 1
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %96, ptr %97, ptr @C_L1L2_7_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %98 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %99 = add i32 %98, 1
  %100 = icmp sge i32 %99, 2
  %101 = add i32 %98, -1
  %102 = select i1 %100, i32 %101, i32 %99
  store i32 %102, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %103 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %104 = add i32 %103, 1
  %105 = icmp sge i32 %104, 2
  %106 = add i32 %103, -1
  %107 = select i1 %105, i32 %106, i32 %104
  store i32 %107, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %108 = add i64 %83, 1
  br label %82

109:                                              ; preds = %82
  %110 = add i64 %80, 1
  br label %79

111:                                              ; preds = %79
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %112 = load i32, ptr @_anonymous7, align 4
  %113 = add i32 %112, 1
  %114 = icmp sge i32 %113, 1
  %115 = select i1 %114, i32 %112, i32 %113
  store i32 %115, ptr @_anonymous7, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_7_0_buff_0)
  br label %116

116:                                              ; preds = %146, %111
  %117 = phi i64 [ %147, %146 ], [ 0, %111 ]
  %118 = icmp slt i64 %117, 2
  br i1 %118, label %119, label %148

119:                                              ; preds = %131, %116
  %120 = phi i64 [ %145, %131 ], [ 0, %116 ]
  %121 = icmp slt i64 %120, 8
  br i1 %121, label %122, label %146

122:                                              ; preds = %119
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %123 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  switch i32 %123, label %124 [
    i32 0, label %124
    i32 1, label %125
  ]

124:                                              ; preds = %122, %122
  br label %126

125:                                              ; preds = %122
  br label %126

126:                                              ; preds = %125, %124
  %127 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %125 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_7_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %124 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %128 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %128, label %129 [
    i32 0, label %129
    i32 1, label %130
  ]

129:                                              ; preds = %126, %126
  br label %131

130:                                              ; preds = %126
  br label %131

131:                                              ; preds = %130, %129
  %132 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %130 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %129 ]
  %133 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, 1
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %133, ptr %134, ptr @C_L1L2_7_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %135 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %136 = add i32 %135, 1
  %137 = icmp sge i32 %136, 2
  %138 = add i32 %135, -1
  %139 = select i1 %137, i32 %138, i32 %136
  store i32 %139, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %140 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %141 = add i32 %140, 1
  %142 = icmp sge i32 %141, 2
  %143 = add i32 %140, -1
  %144 = select i1 %142, i32 %143, i32 %141
  store i32 %144, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %145 = add i64 %120, 1
  br label %119

146:                                              ; preds = %119
  %147 = add i64 %117, 1
  br label %116

148:                                              ; preds = %116
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %149 = load i32, ptr @_anonymous7, align 4
  %150 = add i32 %149, 1
  %151 = icmp sge i32 %150, 1
  %152 = select i1 %151, i32 %149, i32 %150
  store i32 %152, ptr @_anonymous7, align 4
  %153 = add i64 %2, 1
  br label %1

154:                                              ; preds = %1
  ret void
}

declare void @llvm.aie2p.acquire(i32, i32) addrspace(0)

declare void @zero_scalar_i16(ptr) addrspace(0)

declare void @matmul_scalar_i8_i16(ptr, ptr, ptr) addrspace(0)

declare void @llvm.aie2p.release(i32, i32) addrspace(0)

define void @core_6_2() addrspace(0) {
  store i32 0, ptr @_anonymous6, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  br label %1

1:                                                ; preds = %148, %0
  %2 = phi i64 [ %153, %148 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %154

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_6_0_buff_0)
  br label %5

5:                                                ; preds = %35, %4
  %6 = phi i64 [ %36, %35 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 2
  br i1 %7, label %8, label %37

8:                                                ; preds = %20, %5
  %9 = phi i64 [ %34, %20 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 8
  br i1 %10, label %11, label %35

11:                                               ; preds = %8
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %12 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  switch i32 %12, label %13 [
    i32 0, label %13
    i32 1, label %14
  ]

13:                                               ; preds = %11, %11
  br label %15

14:                                               ; preds = %11
  br label %15

15:                                               ; preds = %14, %13
  %16 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %14 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %13 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %17 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %17, label %18 [
    i32 0, label %18
    i32 1, label %19
  ]

18:                                               ; preds = %15, %15
  br label %20

19:                                               ; preds = %15
  br label %20

20:                                               ; preds = %19, %18
  %21 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %18 ]
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %22, ptr %23, ptr @C_L1L2_6_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %24 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %25 = add i32 %24, 1
  %26 = icmp sge i32 %25, 2
  %27 = add i32 %24, -1
  %28 = select i1 %26, i32 %27, i32 %25
  store i32 %28, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %29 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %30 = add i32 %29, 1
  %31 = icmp sge i32 %30, 2
  %32 = add i32 %29, -1
  %33 = select i1 %31, i32 %32, i32 %30
  store i32 %33, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %34 = add i64 %9, 1
  br label %8

35:                                               ; preds = %8
  %36 = add i64 %6, 1
  br label %5

37:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %38 = load i32, ptr @_anonymous6, align 4
  %39 = add i32 %38, 1
  %40 = icmp sge i32 %39, 1
  %41 = select i1 %40, i32 %38, i32 %39
  store i32 %41, ptr @_anonymous6, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_6_0_buff_0)
  br label %42

42:                                               ; preds = %72, %37
  %43 = phi i64 [ %73, %72 ], [ 0, %37 ]
  %44 = icmp slt i64 %43, 2
  br i1 %44, label %45, label %74

45:                                               ; preds = %57, %42
  %46 = phi i64 [ %71, %57 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 8
  br i1 %47, label %48, label %72

48:                                               ; preds = %45
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  switch i32 %49, label %50 [
    i32 0, label %50
    i32 1, label %51
  ]

50:                                               ; preds = %48, %48
  br label %52

51:                                               ; preds = %48
  br label %52

52:                                               ; preds = %51, %50
  %53 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %51 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %50 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %54, label %55 [
    i32 0, label %55
    i32 1, label %56
  ]

55:                                               ; preds = %52, %52
  br label %57

56:                                               ; preds = %52
  br label %57

57:                                               ; preds = %56, %55
  %58 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %56 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %55 ]
  %59 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %59, ptr %60, ptr @C_L1L2_6_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %61 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %62 = add i32 %61, 1
  %63 = icmp sge i32 %62, 2
  %64 = add i32 %61, -1
  %65 = select i1 %63, i32 %64, i32 %62
  store i32 %65, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %66 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 2
  %69 = add i32 %66, -1
  %70 = select i1 %68, i32 %69, i32 %67
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %71 = add i64 %46, 1
  br label %45

72:                                               ; preds = %45
  %73 = add i64 %43, 1
  br label %42

74:                                               ; preds = %42
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %75 = load i32, ptr @_anonymous6, align 4
  %76 = add i32 %75, 1
  %77 = icmp sge i32 %76, 1
  %78 = select i1 %77, i32 %75, i32 %76
  store i32 %78, ptr @_anonymous6, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_6_0_buff_0)
  br label %79

79:                                               ; preds = %109, %74
  %80 = phi i64 [ %110, %109 ], [ 0, %74 ]
  %81 = icmp slt i64 %80, 2
  br i1 %81, label %82, label %111

82:                                               ; preds = %94, %79
  %83 = phi i64 [ %108, %94 ], [ 0, %79 ]
  %84 = icmp slt i64 %83, 8
  br i1 %84, label %85, label %109

85:                                               ; preds = %82
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %86 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  switch i32 %86, label %87 [
    i32 0, label %87
    i32 1, label %88
  ]

87:                                               ; preds = %85, %85
  br label %89

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %88, %87
  %90 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %88 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %87 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %91 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %91, label %92 [
    i32 0, label %92
    i32 1, label %93
  ]

92:                                               ; preds = %89, %89
  br label %94

93:                                               ; preds = %89
  br label %94

94:                                               ; preds = %93, %92
  %95 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %93 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %92 ]
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, 1
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %96, ptr %97, ptr @C_L1L2_6_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %98 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %99 = add i32 %98, 1
  %100 = icmp sge i32 %99, 2
  %101 = add i32 %98, -1
  %102 = select i1 %100, i32 %101, i32 %99
  store i32 %102, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %103 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %104 = add i32 %103, 1
  %105 = icmp sge i32 %104, 2
  %106 = add i32 %103, -1
  %107 = select i1 %105, i32 %106, i32 %104
  store i32 %107, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %108 = add i64 %83, 1
  br label %82

109:                                              ; preds = %82
  %110 = add i64 %80, 1
  br label %79

111:                                              ; preds = %79
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %112 = load i32, ptr @_anonymous6, align 4
  %113 = add i32 %112, 1
  %114 = icmp sge i32 %113, 1
  %115 = select i1 %114, i32 %112, i32 %113
  store i32 %115, ptr @_anonymous6, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_6_0_buff_0)
  br label %116

116:                                              ; preds = %146, %111
  %117 = phi i64 [ %147, %146 ], [ 0, %111 ]
  %118 = icmp slt i64 %117, 2
  br i1 %118, label %119, label %148

119:                                              ; preds = %131, %116
  %120 = phi i64 [ %145, %131 ], [ 0, %116 ]
  %121 = icmp slt i64 %120, 8
  br i1 %121, label %122, label %146

122:                                              ; preds = %119
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %123 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  switch i32 %123, label %124 [
    i32 0, label %124
    i32 1, label %125
  ]

124:                                              ; preds = %122, %122
  br label %126

125:                                              ; preds = %122
  br label %126

126:                                              ; preds = %125, %124
  %127 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %125 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_6_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %124 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %128 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %128, label %129 [
    i32 0, label %129
    i32 1, label %130
  ]

129:                                              ; preds = %126, %126
  br label %131

130:                                              ; preds = %126
  br label %131

131:                                              ; preds = %130, %129
  %132 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %130 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %129 ]
  %133 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, 1
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %133, ptr %134, ptr @C_L1L2_6_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %135 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %136 = add i32 %135, 1
  %137 = icmp sge i32 %136, 2
  %138 = add i32 %135, -1
  %139 = select i1 %137, i32 %138, i32 %136
  store i32 %139, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %140 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %141 = add i32 %140, 1
  %142 = icmp sge i32 %141, 2
  %143 = add i32 %140, -1
  %144 = select i1 %142, i32 %143, i32 %141
  store i32 %144, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %145 = add i64 %120, 1
  br label %119

146:                                              ; preds = %119
  %147 = add i64 %117, 1
  br label %116

148:                                              ; preds = %116
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %149 = load i32, ptr @_anonymous6, align 4
  %150 = add i32 %149, 1
  %151 = icmp sge i32 %150, 1
  %152 = select i1 %151, i32 %149, i32 %150
  store i32 %152, ptr @_anonymous6, align 4
  %153 = add i64 %2, 1
  br label %1

154:                                              ; preds = %1
  ret void
}

define void @core_5_2() addrspace(0) {
  store i32 0, ptr @_anonymous5, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  br label %1

1:                                                ; preds = %148, %0
  %2 = phi i64 [ %153, %148 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %154

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_5_0_buff_0)
  br label %5

5:                                                ; preds = %35, %4
  %6 = phi i64 [ %36, %35 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 2
  br i1 %7, label %8, label %37

8:                                                ; preds = %20, %5
  %9 = phi i64 [ %34, %20 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 8
  br i1 %10, label %11, label %35

11:                                               ; preds = %8
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %12 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  switch i32 %12, label %13 [
    i32 0, label %13
    i32 1, label %14
  ]

13:                                               ; preds = %11, %11
  br label %15

14:                                               ; preds = %11
  br label %15

15:                                               ; preds = %14, %13
  %16 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %14 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %13 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %17 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %17, label %18 [
    i32 0, label %18
    i32 1, label %19
  ]

18:                                               ; preds = %15, %15
  br label %20

19:                                               ; preds = %15
  br label %20

20:                                               ; preds = %19, %18
  %21 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %18 ]
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %22, ptr %23, ptr @C_L1L2_5_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %24 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %25 = add i32 %24, 1
  %26 = icmp sge i32 %25, 2
  %27 = add i32 %24, -1
  %28 = select i1 %26, i32 %27, i32 %25
  store i32 %28, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %29 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %30 = add i32 %29, 1
  %31 = icmp sge i32 %30, 2
  %32 = add i32 %29, -1
  %33 = select i1 %31, i32 %32, i32 %30
  store i32 %33, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %34 = add i64 %9, 1
  br label %8

35:                                               ; preds = %8
  %36 = add i64 %6, 1
  br label %5

37:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %38 = load i32, ptr @_anonymous5, align 4
  %39 = add i32 %38, 1
  %40 = icmp sge i32 %39, 1
  %41 = select i1 %40, i32 %38, i32 %39
  store i32 %41, ptr @_anonymous5, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_5_0_buff_0)
  br label %42

42:                                               ; preds = %72, %37
  %43 = phi i64 [ %73, %72 ], [ 0, %37 ]
  %44 = icmp slt i64 %43, 2
  br i1 %44, label %45, label %74

45:                                               ; preds = %57, %42
  %46 = phi i64 [ %71, %57 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 8
  br i1 %47, label %48, label %72

48:                                               ; preds = %45
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  switch i32 %49, label %50 [
    i32 0, label %50
    i32 1, label %51
  ]

50:                                               ; preds = %48, %48
  br label %52

51:                                               ; preds = %48
  br label %52

52:                                               ; preds = %51, %50
  %53 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %51 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %50 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %54, label %55 [
    i32 0, label %55
    i32 1, label %56
  ]

55:                                               ; preds = %52, %52
  br label %57

56:                                               ; preds = %52
  br label %57

57:                                               ; preds = %56, %55
  %58 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %56 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %55 ]
  %59 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %59, ptr %60, ptr @C_L1L2_5_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %61 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %62 = add i32 %61, 1
  %63 = icmp sge i32 %62, 2
  %64 = add i32 %61, -1
  %65 = select i1 %63, i32 %64, i32 %62
  store i32 %65, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %66 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 2
  %69 = add i32 %66, -1
  %70 = select i1 %68, i32 %69, i32 %67
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %71 = add i64 %46, 1
  br label %45

72:                                               ; preds = %45
  %73 = add i64 %43, 1
  br label %42

74:                                               ; preds = %42
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %75 = load i32, ptr @_anonymous5, align 4
  %76 = add i32 %75, 1
  %77 = icmp sge i32 %76, 1
  %78 = select i1 %77, i32 %75, i32 %76
  store i32 %78, ptr @_anonymous5, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_5_0_buff_0)
  br label %79

79:                                               ; preds = %109, %74
  %80 = phi i64 [ %110, %109 ], [ 0, %74 ]
  %81 = icmp slt i64 %80, 2
  br i1 %81, label %82, label %111

82:                                               ; preds = %94, %79
  %83 = phi i64 [ %108, %94 ], [ 0, %79 ]
  %84 = icmp slt i64 %83, 8
  br i1 %84, label %85, label %109

85:                                               ; preds = %82
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %86 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  switch i32 %86, label %87 [
    i32 0, label %87
    i32 1, label %88
  ]

87:                                               ; preds = %85, %85
  br label %89

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %88, %87
  %90 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %88 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %87 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %91 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %91, label %92 [
    i32 0, label %92
    i32 1, label %93
  ]

92:                                               ; preds = %89, %89
  br label %94

93:                                               ; preds = %89
  br label %94

94:                                               ; preds = %93, %92
  %95 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %93 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %92 ]
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, 1
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %96, ptr %97, ptr @C_L1L2_5_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %98 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %99 = add i32 %98, 1
  %100 = icmp sge i32 %99, 2
  %101 = add i32 %98, -1
  %102 = select i1 %100, i32 %101, i32 %99
  store i32 %102, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %103 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %104 = add i32 %103, 1
  %105 = icmp sge i32 %104, 2
  %106 = add i32 %103, -1
  %107 = select i1 %105, i32 %106, i32 %104
  store i32 %107, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %108 = add i64 %83, 1
  br label %82

109:                                              ; preds = %82
  %110 = add i64 %80, 1
  br label %79

111:                                              ; preds = %79
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %112 = load i32, ptr @_anonymous5, align 4
  %113 = add i32 %112, 1
  %114 = icmp sge i32 %113, 1
  %115 = select i1 %114, i32 %112, i32 %113
  store i32 %115, ptr @_anonymous5, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_5_0_buff_0)
  br label %116

116:                                              ; preds = %146, %111
  %117 = phi i64 [ %147, %146 ], [ 0, %111 ]
  %118 = icmp slt i64 %117, 2
  br i1 %118, label %119, label %148

119:                                              ; preds = %131, %116
  %120 = phi i64 [ %145, %131 ], [ 0, %116 ]
  %121 = icmp slt i64 %120, 8
  br i1 %121, label %122, label %146

122:                                              ; preds = %119
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %123 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  switch i32 %123, label %124 [
    i32 0, label %124
    i32 1, label %125
  ]

124:                                              ; preds = %122, %122
  br label %126

125:                                              ; preds = %122
  br label %126

126:                                              ; preds = %125, %124
  %127 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %125 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_5_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %124 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %128 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %128, label %129 [
    i32 0, label %129
    i32 1, label %130
  ]

129:                                              ; preds = %126, %126
  br label %131

130:                                              ; preds = %126
  br label %131

131:                                              ; preds = %130, %129
  %132 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %130 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %129 ]
  %133 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, 1
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %133, ptr %134, ptr @C_L1L2_5_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %135 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %136 = add i32 %135, 1
  %137 = icmp sge i32 %136, 2
  %138 = add i32 %135, -1
  %139 = select i1 %137, i32 %138, i32 %136
  store i32 %139, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %140 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %141 = add i32 %140, 1
  %142 = icmp sge i32 %141, 2
  %143 = add i32 %140, -1
  %144 = select i1 %142, i32 %143, i32 %141
  store i32 %144, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %145 = add i64 %120, 1
  br label %119

146:                                              ; preds = %119
  %147 = add i64 %117, 1
  br label %116

148:                                              ; preds = %116
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %149 = load i32, ptr @_anonymous5, align 4
  %150 = add i32 %149, 1
  %151 = icmp sge i32 %150, 1
  %152 = select i1 %151, i32 %149, i32 %150
  store i32 %152, ptr @_anonymous5, align 4
  %153 = add i64 %2, 1
  br label %1

154:                                              ; preds = %1
  ret void
}

define void @core_4_2() addrspace(0) {
  store i32 0, ptr @_anonymous4, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  br label %1

1:                                                ; preds = %148, %0
  %2 = phi i64 [ %153, %148 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %154

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_4_0_buff_0)
  br label %5

5:                                                ; preds = %35, %4
  %6 = phi i64 [ %36, %35 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 2
  br i1 %7, label %8, label %37

8:                                                ; preds = %20, %5
  %9 = phi i64 [ %34, %20 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 8
  br i1 %10, label %11, label %35

11:                                               ; preds = %8
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %12 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %12, label %13 [
    i32 0, label %13
    i32 1, label %14
  ]

13:                                               ; preds = %11, %11
  br label %15

14:                                               ; preds = %11
  br label %15

15:                                               ; preds = %14, %13
  %16 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %14 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %13 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %17 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %17, label %18 [
    i32 0, label %18
    i32 1, label %19
  ]

18:                                               ; preds = %15, %15
  br label %20

19:                                               ; preds = %15
  br label %20

20:                                               ; preds = %19, %18
  %21 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %18 ]
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %22, ptr %23, ptr @C_L1L2_4_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %24 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %25 = add i32 %24, 1
  %26 = icmp sge i32 %25, 2
  %27 = add i32 %24, -1
  %28 = select i1 %26, i32 %27, i32 %25
  store i32 %28, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %29 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %30 = add i32 %29, 1
  %31 = icmp sge i32 %30, 2
  %32 = add i32 %29, -1
  %33 = select i1 %31, i32 %32, i32 %30
  store i32 %33, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %34 = add i64 %9, 1
  br label %8

35:                                               ; preds = %8
  %36 = add i64 %6, 1
  br label %5

37:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %38 = load i32, ptr @_anonymous4, align 4
  %39 = add i32 %38, 1
  %40 = icmp sge i32 %39, 1
  %41 = select i1 %40, i32 %38, i32 %39
  store i32 %41, ptr @_anonymous4, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_4_0_buff_0)
  br label %42

42:                                               ; preds = %72, %37
  %43 = phi i64 [ %73, %72 ], [ 0, %37 ]
  %44 = icmp slt i64 %43, 2
  br i1 %44, label %45, label %74

45:                                               ; preds = %57, %42
  %46 = phi i64 [ %71, %57 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 8
  br i1 %47, label %48, label %72

48:                                               ; preds = %45
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %49, label %50 [
    i32 0, label %50
    i32 1, label %51
  ]

50:                                               ; preds = %48, %48
  br label %52

51:                                               ; preds = %48
  br label %52

52:                                               ; preds = %51, %50
  %53 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %51 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %50 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %54, label %55 [
    i32 0, label %55
    i32 1, label %56
  ]

55:                                               ; preds = %52, %52
  br label %57

56:                                               ; preds = %52
  br label %57

57:                                               ; preds = %56, %55
  %58 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %56 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %55 ]
  %59 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %59, ptr %60, ptr @C_L1L2_4_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %61 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %62 = add i32 %61, 1
  %63 = icmp sge i32 %62, 2
  %64 = add i32 %61, -1
  %65 = select i1 %63, i32 %64, i32 %62
  store i32 %65, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %66 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 2
  %69 = add i32 %66, -1
  %70 = select i1 %68, i32 %69, i32 %67
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %71 = add i64 %46, 1
  br label %45

72:                                               ; preds = %45
  %73 = add i64 %43, 1
  br label %42

74:                                               ; preds = %42
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %75 = load i32, ptr @_anonymous4, align 4
  %76 = add i32 %75, 1
  %77 = icmp sge i32 %76, 1
  %78 = select i1 %77, i32 %75, i32 %76
  store i32 %78, ptr @_anonymous4, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_4_0_buff_0)
  br label %79

79:                                               ; preds = %109, %74
  %80 = phi i64 [ %110, %109 ], [ 0, %74 ]
  %81 = icmp slt i64 %80, 2
  br i1 %81, label %82, label %111

82:                                               ; preds = %94, %79
  %83 = phi i64 [ %108, %94 ], [ 0, %79 ]
  %84 = icmp slt i64 %83, 8
  br i1 %84, label %85, label %109

85:                                               ; preds = %82
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %86 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %86, label %87 [
    i32 0, label %87
    i32 1, label %88
  ]

87:                                               ; preds = %85, %85
  br label %89

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %88, %87
  %90 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %88 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %87 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %91 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %91, label %92 [
    i32 0, label %92
    i32 1, label %93
  ]

92:                                               ; preds = %89, %89
  br label %94

93:                                               ; preds = %89
  br label %94

94:                                               ; preds = %93, %92
  %95 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %93 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %92 ]
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, 1
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %96, ptr %97, ptr @C_L1L2_4_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %98 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %99 = add i32 %98, 1
  %100 = icmp sge i32 %99, 2
  %101 = add i32 %98, -1
  %102 = select i1 %100, i32 %101, i32 %99
  store i32 %102, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %103 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %104 = add i32 %103, 1
  %105 = icmp sge i32 %104, 2
  %106 = add i32 %103, -1
  %107 = select i1 %105, i32 %106, i32 %104
  store i32 %107, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %108 = add i64 %83, 1
  br label %82

109:                                              ; preds = %82
  %110 = add i64 %80, 1
  br label %79

111:                                              ; preds = %79
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %112 = load i32, ptr @_anonymous4, align 4
  %113 = add i32 %112, 1
  %114 = icmp sge i32 %113, 1
  %115 = select i1 %114, i32 %112, i32 %113
  store i32 %115, ptr @_anonymous4, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_4_0_buff_0)
  br label %116

116:                                              ; preds = %146, %111
  %117 = phi i64 [ %147, %146 ], [ 0, %111 ]
  %118 = icmp slt i64 %117, 2
  br i1 %118, label %119, label %148

119:                                              ; preds = %131, %116
  %120 = phi i64 [ %145, %131 ], [ 0, %116 ]
  %121 = icmp slt i64 %120, 8
  br i1 %121, label %122, label %146

122:                                              ; preds = %119
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %123 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %123, label %124 [
    i32 0, label %124
    i32 1, label %125
  ]

124:                                              ; preds = %122, %122
  br label %126

125:                                              ; preds = %122
  br label %126

126:                                              ; preds = %125, %124
  %127 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %125 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_4_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %124 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %128 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %128, label %129 [
    i32 0, label %129
    i32 1, label %130
  ]

129:                                              ; preds = %126, %126
  br label %131

130:                                              ; preds = %126
  br label %131

131:                                              ; preds = %130, %129
  %132 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %130 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %129 ]
  %133 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, 1
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %133, ptr %134, ptr @C_L1L2_4_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %135 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %136 = add i32 %135, 1
  %137 = icmp sge i32 %136, 2
  %138 = add i32 %135, -1
  %139 = select i1 %137, i32 %138, i32 %136
  store i32 %139, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %140 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %141 = add i32 %140, 1
  %142 = icmp sge i32 %141, 2
  %143 = add i32 %140, -1
  %144 = select i1 %142, i32 %143, i32 %141
  store i32 %144, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %145 = add i64 %120, 1
  br label %119

146:                                              ; preds = %119
  %147 = add i64 %117, 1
  br label %116

148:                                              ; preds = %116
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %149 = load i32, ptr @_anonymous4, align 4
  %150 = add i32 %149, 1
  %151 = icmp sge i32 %150, 1
  %152 = select i1 %151, i32 %149, i32 %150
  store i32 %152, ptr @_anonymous4, align 4
  %153 = add i64 %2, 1
  br label %1

154:                                              ; preds = %1
  ret void
}

define void @core_3_2() addrspace(0) {
  store i32 0, ptr @_anonymous3, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  br label %1

1:                                                ; preds = %148, %0
  %2 = phi i64 [ %153, %148 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %154

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_3_0_buff_0)
  br label %5

5:                                                ; preds = %35, %4
  %6 = phi i64 [ %36, %35 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 2
  br i1 %7, label %8, label %37

8:                                                ; preds = %20, %5
  %9 = phi i64 [ %34, %20 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 8
  br i1 %10, label %11, label %35

11:                                               ; preds = %8
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %12 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  switch i32 %12, label %13 [
    i32 0, label %13
    i32 1, label %14
  ]

13:                                               ; preds = %11, %11
  br label %15

14:                                               ; preds = %11
  br label %15

15:                                               ; preds = %14, %13
  %16 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %14 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %13 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %17 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %17, label %18 [
    i32 0, label %18
    i32 1, label %19
  ]

18:                                               ; preds = %15, %15
  br label %20

19:                                               ; preds = %15
  br label %20

20:                                               ; preds = %19, %18
  %21 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %18 ]
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %22, ptr %23, ptr @C_L1L2_3_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %24 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %25 = add i32 %24, 1
  %26 = icmp sge i32 %25, 2
  %27 = add i32 %24, -1
  %28 = select i1 %26, i32 %27, i32 %25
  store i32 %28, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %29 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %30 = add i32 %29, 1
  %31 = icmp sge i32 %30, 2
  %32 = add i32 %29, -1
  %33 = select i1 %31, i32 %32, i32 %30
  store i32 %33, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %34 = add i64 %9, 1
  br label %8

35:                                               ; preds = %8
  %36 = add i64 %6, 1
  br label %5

37:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %38 = load i32, ptr @_anonymous3, align 4
  %39 = add i32 %38, 1
  %40 = icmp sge i32 %39, 1
  %41 = select i1 %40, i32 %38, i32 %39
  store i32 %41, ptr @_anonymous3, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_3_0_buff_0)
  br label %42

42:                                               ; preds = %72, %37
  %43 = phi i64 [ %73, %72 ], [ 0, %37 ]
  %44 = icmp slt i64 %43, 2
  br i1 %44, label %45, label %74

45:                                               ; preds = %57, %42
  %46 = phi i64 [ %71, %57 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 8
  br i1 %47, label %48, label %72

48:                                               ; preds = %45
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  switch i32 %49, label %50 [
    i32 0, label %50
    i32 1, label %51
  ]

50:                                               ; preds = %48, %48
  br label %52

51:                                               ; preds = %48
  br label %52

52:                                               ; preds = %51, %50
  %53 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %51 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %50 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %54, label %55 [
    i32 0, label %55
    i32 1, label %56
  ]

55:                                               ; preds = %52, %52
  br label %57

56:                                               ; preds = %52
  br label %57

57:                                               ; preds = %56, %55
  %58 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %56 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %55 ]
  %59 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %59, ptr %60, ptr @C_L1L2_3_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %61 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %62 = add i32 %61, 1
  %63 = icmp sge i32 %62, 2
  %64 = add i32 %61, -1
  %65 = select i1 %63, i32 %64, i32 %62
  store i32 %65, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %66 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 2
  %69 = add i32 %66, -1
  %70 = select i1 %68, i32 %69, i32 %67
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %71 = add i64 %46, 1
  br label %45

72:                                               ; preds = %45
  %73 = add i64 %43, 1
  br label %42

74:                                               ; preds = %42
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %75 = load i32, ptr @_anonymous3, align 4
  %76 = add i32 %75, 1
  %77 = icmp sge i32 %76, 1
  %78 = select i1 %77, i32 %75, i32 %76
  store i32 %78, ptr @_anonymous3, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_3_0_buff_0)
  br label %79

79:                                               ; preds = %109, %74
  %80 = phi i64 [ %110, %109 ], [ 0, %74 ]
  %81 = icmp slt i64 %80, 2
  br i1 %81, label %82, label %111

82:                                               ; preds = %94, %79
  %83 = phi i64 [ %108, %94 ], [ 0, %79 ]
  %84 = icmp slt i64 %83, 8
  br i1 %84, label %85, label %109

85:                                               ; preds = %82
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %86 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  switch i32 %86, label %87 [
    i32 0, label %87
    i32 1, label %88
  ]

87:                                               ; preds = %85, %85
  br label %89

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %88, %87
  %90 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %88 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %87 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %91 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %91, label %92 [
    i32 0, label %92
    i32 1, label %93
  ]

92:                                               ; preds = %89, %89
  br label %94

93:                                               ; preds = %89
  br label %94

94:                                               ; preds = %93, %92
  %95 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %93 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %92 ]
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, 1
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %96, ptr %97, ptr @C_L1L2_3_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %98 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %99 = add i32 %98, 1
  %100 = icmp sge i32 %99, 2
  %101 = add i32 %98, -1
  %102 = select i1 %100, i32 %101, i32 %99
  store i32 %102, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %103 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %104 = add i32 %103, 1
  %105 = icmp sge i32 %104, 2
  %106 = add i32 %103, -1
  %107 = select i1 %105, i32 %106, i32 %104
  store i32 %107, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %108 = add i64 %83, 1
  br label %82

109:                                              ; preds = %82
  %110 = add i64 %80, 1
  br label %79

111:                                              ; preds = %79
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %112 = load i32, ptr @_anonymous3, align 4
  %113 = add i32 %112, 1
  %114 = icmp sge i32 %113, 1
  %115 = select i1 %114, i32 %112, i32 %113
  store i32 %115, ptr @_anonymous3, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_3_0_buff_0)
  br label %116

116:                                              ; preds = %146, %111
  %117 = phi i64 [ %147, %146 ], [ 0, %111 ]
  %118 = icmp slt i64 %117, 2
  br i1 %118, label %119, label %148

119:                                              ; preds = %131, %116
  %120 = phi i64 [ %145, %131 ], [ 0, %116 ]
  %121 = icmp slt i64 %120, 8
  br i1 %121, label %122, label %146

122:                                              ; preds = %119
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %123 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  switch i32 %123, label %124 [
    i32 0, label %124
    i32 1, label %125
  ]

124:                                              ; preds = %122, %122
  br label %126

125:                                              ; preds = %122
  br label %126

126:                                              ; preds = %125, %124
  %127 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %125 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_3_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %124 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %128 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %128, label %129 [
    i32 0, label %129
    i32 1, label %130
  ]

129:                                              ; preds = %126, %126
  br label %131

130:                                              ; preds = %126
  br label %131

131:                                              ; preds = %130, %129
  %132 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %130 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %129 ]
  %133 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, 1
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %133, ptr %134, ptr @C_L1L2_3_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %135 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %136 = add i32 %135, 1
  %137 = icmp sge i32 %136, 2
  %138 = add i32 %135, -1
  %139 = select i1 %137, i32 %138, i32 %136
  store i32 %139, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %140 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %141 = add i32 %140, 1
  %142 = icmp sge i32 %141, 2
  %143 = add i32 %140, -1
  %144 = select i1 %142, i32 %143, i32 %141
  store i32 %144, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %145 = add i64 %120, 1
  br label %119

146:                                              ; preds = %119
  %147 = add i64 %117, 1
  br label %116

148:                                              ; preds = %116
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %149 = load i32, ptr @_anonymous3, align 4
  %150 = add i32 %149, 1
  %151 = icmp sge i32 %150, 1
  %152 = select i1 %151, i32 %149, i32 %150
  store i32 %152, ptr @_anonymous3, align 4
  %153 = add i64 %2, 1
  br label %1

154:                                              ; preds = %1
  ret void
}

define void @core_2_2() addrspace(0) {
  store i32 0, ptr @_anonymous2, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  br label %1

1:                                                ; preds = %148, %0
  %2 = phi i64 [ %153, %148 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %154

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_2_0_buff_0)
  br label %5

5:                                                ; preds = %35, %4
  %6 = phi i64 [ %36, %35 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 2
  br i1 %7, label %8, label %37

8:                                                ; preds = %20, %5
  %9 = phi i64 [ %34, %20 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 8
  br i1 %10, label %11, label %35

11:                                               ; preds = %8
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %12 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  switch i32 %12, label %13 [
    i32 0, label %13
    i32 1, label %14
  ]

13:                                               ; preds = %11, %11
  br label %15

14:                                               ; preds = %11
  br label %15

15:                                               ; preds = %14, %13
  %16 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %14 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %13 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %17 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %17, label %18 [
    i32 0, label %18
    i32 1, label %19
  ]

18:                                               ; preds = %15, %15
  br label %20

19:                                               ; preds = %15
  br label %20

20:                                               ; preds = %19, %18
  %21 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %18 ]
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %22, ptr %23, ptr @C_L1L2_2_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %24 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %25 = add i32 %24, 1
  %26 = icmp sge i32 %25, 2
  %27 = add i32 %24, -1
  %28 = select i1 %26, i32 %27, i32 %25
  store i32 %28, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %29 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %30 = add i32 %29, 1
  %31 = icmp sge i32 %30, 2
  %32 = add i32 %29, -1
  %33 = select i1 %31, i32 %32, i32 %30
  store i32 %33, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %34 = add i64 %9, 1
  br label %8

35:                                               ; preds = %8
  %36 = add i64 %6, 1
  br label %5

37:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %38 = load i32, ptr @_anonymous2, align 4
  %39 = add i32 %38, 1
  %40 = icmp sge i32 %39, 1
  %41 = select i1 %40, i32 %38, i32 %39
  store i32 %41, ptr @_anonymous2, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_2_0_buff_0)
  br label %42

42:                                               ; preds = %72, %37
  %43 = phi i64 [ %73, %72 ], [ 0, %37 ]
  %44 = icmp slt i64 %43, 2
  br i1 %44, label %45, label %74

45:                                               ; preds = %57, %42
  %46 = phi i64 [ %71, %57 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 8
  br i1 %47, label %48, label %72

48:                                               ; preds = %45
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  switch i32 %49, label %50 [
    i32 0, label %50
    i32 1, label %51
  ]

50:                                               ; preds = %48, %48
  br label %52

51:                                               ; preds = %48
  br label %52

52:                                               ; preds = %51, %50
  %53 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %51 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %50 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %54, label %55 [
    i32 0, label %55
    i32 1, label %56
  ]

55:                                               ; preds = %52, %52
  br label %57

56:                                               ; preds = %52
  br label %57

57:                                               ; preds = %56, %55
  %58 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %56 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %55 ]
  %59 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %59, ptr %60, ptr @C_L1L2_2_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %61 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %62 = add i32 %61, 1
  %63 = icmp sge i32 %62, 2
  %64 = add i32 %61, -1
  %65 = select i1 %63, i32 %64, i32 %62
  store i32 %65, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %66 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 2
  %69 = add i32 %66, -1
  %70 = select i1 %68, i32 %69, i32 %67
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %71 = add i64 %46, 1
  br label %45

72:                                               ; preds = %45
  %73 = add i64 %43, 1
  br label %42

74:                                               ; preds = %42
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %75 = load i32, ptr @_anonymous2, align 4
  %76 = add i32 %75, 1
  %77 = icmp sge i32 %76, 1
  %78 = select i1 %77, i32 %75, i32 %76
  store i32 %78, ptr @_anonymous2, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_2_0_buff_0)
  br label %79

79:                                               ; preds = %109, %74
  %80 = phi i64 [ %110, %109 ], [ 0, %74 ]
  %81 = icmp slt i64 %80, 2
  br i1 %81, label %82, label %111

82:                                               ; preds = %94, %79
  %83 = phi i64 [ %108, %94 ], [ 0, %79 ]
  %84 = icmp slt i64 %83, 8
  br i1 %84, label %85, label %109

85:                                               ; preds = %82
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %86 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  switch i32 %86, label %87 [
    i32 0, label %87
    i32 1, label %88
  ]

87:                                               ; preds = %85, %85
  br label %89

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %88, %87
  %90 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %88 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %87 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %91 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %91, label %92 [
    i32 0, label %92
    i32 1, label %93
  ]

92:                                               ; preds = %89, %89
  br label %94

93:                                               ; preds = %89
  br label %94

94:                                               ; preds = %93, %92
  %95 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %93 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %92 ]
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, 1
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %96, ptr %97, ptr @C_L1L2_2_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %98 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %99 = add i32 %98, 1
  %100 = icmp sge i32 %99, 2
  %101 = add i32 %98, -1
  %102 = select i1 %100, i32 %101, i32 %99
  store i32 %102, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %103 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %104 = add i32 %103, 1
  %105 = icmp sge i32 %104, 2
  %106 = add i32 %103, -1
  %107 = select i1 %105, i32 %106, i32 %104
  store i32 %107, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %108 = add i64 %83, 1
  br label %82

109:                                              ; preds = %82
  %110 = add i64 %80, 1
  br label %79

111:                                              ; preds = %79
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %112 = load i32, ptr @_anonymous2, align 4
  %113 = add i32 %112, 1
  %114 = icmp sge i32 %113, 1
  %115 = select i1 %114, i32 %112, i32 %113
  store i32 %115, ptr @_anonymous2, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_2_0_buff_0)
  br label %116

116:                                              ; preds = %146, %111
  %117 = phi i64 [ %147, %146 ], [ 0, %111 ]
  %118 = icmp slt i64 %117, 2
  br i1 %118, label %119, label %148

119:                                              ; preds = %131, %116
  %120 = phi i64 [ %145, %131 ], [ 0, %116 ]
  %121 = icmp slt i64 %120, 8
  br i1 %121, label %122, label %146

122:                                              ; preds = %119
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %123 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  switch i32 %123, label %124 [
    i32 0, label %124
    i32 1, label %125
  ]

124:                                              ; preds = %122, %122
  br label %126

125:                                              ; preds = %122
  br label %126

126:                                              ; preds = %125, %124
  %127 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %125 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_2_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %124 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %128 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %128, label %129 [
    i32 0, label %129
    i32 1, label %130
  ]

129:                                              ; preds = %126, %126
  br label %131

130:                                              ; preds = %126
  br label %131

131:                                              ; preds = %130, %129
  %132 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %130 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %129 ]
  %133 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, 1
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %133, ptr %134, ptr @C_L1L2_2_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %135 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %136 = add i32 %135, 1
  %137 = icmp sge i32 %136, 2
  %138 = add i32 %135, -1
  %139 = select i1 %137, i32 %138, i32 %136
  store i32 %139, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %140 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %141 = add i32 %140, 1
  %142 = icmp sge i32 %141, 2
  %143 = add i32 %140, -1
  %144 = select i1 %142, i32 %143, i32 %141
  store i32 %144, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %145 = add i64 %120, 1
  br label %119

146:                                              ; preds = %119
  %147 = add i64 %117, 1
  br label %116

148:                                              ; preds = %116
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %149 = load i32, ptr @_anonymous2, align 4
  %150 = add i32 %149, 1
  %151 = icmp sge i32 %150, 1
  %152 = select i1 %151, i32 %149, i32 %150
  store i32 %152, ptr @_anonymous2, align 4
  %153 = add i64 %2, 1
  br label %1

154:                                              ; preds = %1
  ret void
}

define void @core_1_2() addrspace(0) {
  store i32 0, ptr @_anonymous1, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  br label %1

1:                                                ; preds = %148, %0
  %2 = phi i64 [ %153, %148 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %154

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_1_0_buff_0)
  br label %5

5:                                                ; preds = %35, %4
  %6 = phi i64 [ %36, %35 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 2
  br i1 %7, label %8, label %37

8:                                                ; preds = %20, %5
  %9 = phi i64 [ %34, %20 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 8
  br i1 %10, label %11, label %35

11:                                               ; preds = %8
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %12 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  switch i32 %12, label %13 [
    i32 0, label %13
    i32 1, label %14
  ]

13:                                               ; preds = %11, %11
  br label %15

14:                                               ; preds = %11
  br label %15

15:                                               ; preds = %14, %13
  %16 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %14 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %13 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %17 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %17, label %18 [
    i32 0, label %18
    i32 1, label %19
  ]

18:                                               ; preds = %15, %15
  br label %20

19:                                               ; preds = %15
  br label %20

20:                                               ; preds = %19, %18
  %21 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %18 ]
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %22, ptr %23, ptr @C_L1L2_1_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %24 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %25 = add i32 %24, 1
  %26 = icmp sge i32 %25, 2
  %27 = add i32 %24, -1
  %28 = select i1 %26, i32 %27, i32 %25
  store i32 %28, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %29 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %30 = add i32 %29, 1
  %31 = icmp sge i32 %30, 2
  %32 = add i32 %29, -1
  %33 = select i1 %31, i32 %32, i32 %30
  store i32 %33, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %34 = add i64 %9, 1
  br label %8

35:                                               ; preds = %8
  %36 = add i64 %6, 1
  br label %5

37:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %38 = load i32, ptr @_anonymous1, align 4
  %39 = add i32 %38, 1
  %40 = icmp sge i32 %39, 1
  %41 = select i1 %40, i32 %38, i32 %39
  store i32 %41, ptr @_anonymous1, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_1_0_buff_0)
  br label %42

42:                                               ; preds = %72, %37
  %43 = phi i64 [ %73, %72 ], [ 0, %37 ]
  %44 = icmp slt i64 %43, 2
  br i1 %44, label %45, label %74

45:                                               ; preds = %57, %42
  %46 = phi i64 [ %71, %57 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 8
  br i1 %47, label %48, label %72

48:                                               ; preds = %45
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  switch i32 %49, label %50 [
    i32 0, label %50
    i32 1, label %51
  ]

50:                                               ; preds = %48, %48
  br label %52

51:                                               ; preds = %48
  br label %52

52:                                               ; preds = %51, %50
  %53 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %51 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %50 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %54, label %55 [
    i32 0, label %55
    i32 1, label %56
  ]

55:                                               ; preds = %52, %52
  br label %57

56:                                               ; preds = %52
  br label %57

57:                                               ; preds = %56, %55
  %58 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %56 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %55 ]
  %59 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %59, ptr %60, ptr @C_L1L2_1_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %61 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %62 = add i32 %61, 1
  %63 = icmp sge i32 %62, 2
  %64 = add i32 %61, -1
  %65 = select i1 %63, i32 %64, i32 %62
  store i32 %65, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %66 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 2
  %69 = add i32 %66, -1
  %70 = select i1 %68, i32 %69, i32 %67
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %71 = add i64 %46, 1
  br label %45

72:                                               ; preds = %45
  %73 = add i64 %43, 1
  br label %42

74:                                               ; preds = %42
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %75 = load i32, ptr @_anonymous1, align 4
  %76 = add i32 %75, 1
  %77 = icmp sge i32 %76, 1
  %78 = select i1 %77, i32 %75, i32 %76
  store i32 %78, ptr @_anonymous1, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_1_0_buff_0)
  br label %79

79:                                               ; preds = %109, %74
  %80 = phi i64 [ %110, %109 ], [ 0, %74 ]
  %81 = icmp slt i64 %80, 2
  br i1 %81, label %82, label %111

82:                                               ; preds = %94, %79
  %83 = phi i64 [ %108, %94 ], [ 0, %79 ]
  %84 = icmp slt i64 %83, 8
  br i1 %84, label %85, label %109

85:                                               ; preds = %82
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %86 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  switch i32 %86, label %87 [
    i32 0, label %87
    i32 1, label %88
  ]

87:                                               ; preds = %85, %85
  br label %89

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %88, %87
  %90 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %88 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %87 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %91 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %91, label %92 [
    i32 0, label %92
    i32 1, label %93
  ]

92:                                               ; preds = %89, %89
  br label %94

93:                                               ; preds = %89
  br label %94

94:                                               ; preds = %93, %92
  %95 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %93 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %92 ]
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, 1
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %96, ptr %97, ptr @C_L1L2_1_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %98 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %99 = add i32 %98, 1
  %100 = icmp sge i32 %99, 2
  %101 = add i32 %98, -1
  %102 = select i1 %100, i32 %101, i32 %99
  store i32 %102, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %103 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %104 = add i32 %103, 1
  %105 = icmp sge i32 %104, 2
  %106 = add i32 %103, -1
  %107 = select i1 %105, i32 %106, i32 %104
  store i32 %107, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %108 = add i64 %83, 1
  br label %82

109:                                              ; preds = %82
  %110 = add i64 %80, 1
  br label %79

111:                                              ; preds = %79
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %112 = load i32, ptr @_anonymous1, align 4
  %113 = add i32 %112, 1
  %114 = icmp sge i32 %113, 1
  %115 = select i1 %114, i32 %112, i32 %113
  store i32 %115, ptr @_anonymous1, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_1_0_buff_0)
  br label %116

116:                                              ; preds = %146, %111
  %117 = phi i64 [ %147, %146 ], [ 0, %111 ]
  %118 = icmp slt i64 %117, 2
  br i1 %118, label %119, label %148

119:                                              ; preds = %131, %116
  %120 = phi i64 [ %145, %131 ], [ 0, %116 ]
  %121 = icmp slt i64 %120, 8
  br i1 %121, label %122, label %146

122:                                              ; preds = %119
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %123 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  switch i32 %123, label %124 [
    i32 0, label %124
    i32 1, label %125
  ]

124:                                              ; preds = %122, %122
  br label %126

125:                                              ; preds = %122
  br label %126

126:                                              ; preds = %125, %124
  %127 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %125 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %124 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %128 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %128, label %129 [
    i32 0, label %129
    i32 1, label %130
  ]

129:                                              ; preds = %126, %126
  br label %131

130:                                              ; preds = %126
  br label %131

131:                                              ; preds = %130, %129
  %132 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %130 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %129 ]
  %133 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, 1
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %133, ptr %134, ptr @C_L1L2_1_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %135 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %136 = add i32 %135, 1
  %137 = icmp sge i32 %136, 2
  %138 = add i32 %135, -1
  %139 = select i1 %137, i32 %138, i32 %136
  store i32 %139, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %140 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %141 = add i32 %140, 1
  %142 = icmp sge i32 %141, 2
  %143 = add i32 %140, -1
  %144 = select i1 %142, i32 %143, i32 %141
  store i32 %144, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %145 = add i64 %120, 1
  br label %119

146:                                              ; preds = %119
  %147 = add i64 %117, 1
  br label %116

148:                                              ; preds = %116
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %149 = load i32, ptr @_anonymous1, align 4
  %150 = add i32 %149, 1
  %151 = icmp sge i32 %150, 1
  %152 = select i1 %151, i32 %149, i32 %150
  store i32 %152, ptr @_anonymous1, align 4
  %153 = add i64 %2, 1
  br label %1

154:                                              ; preds = %1
  ret void
}

define void @core_0_2() addrspace(0) {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  br label %1

1:                                                ; preds = %148, %0
  %2 = phi i64 [ %153, %148 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %154

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_0_0_buff_0)
  br label %5

5:                                                ; preds = %35, %4
  %6 = phi i64 [ %36, %35 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 2
  br i1 %7, label %8, label %37

8:                                                ; preds = %20, %5
  %9 = phi i64 [ %34, %20 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 8
  br i1 %10, label %11, label %35

11:                                               ; preds = %8
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %12 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %12, label %13 [
    i32 0, label %13
    i32 1, label %14
  ]

13:                                               ; preds = %11, %11
  br label %15

14:                                               ; preds = %11
  br label %15

15:                                               ; preds = %14, %13
  %16 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %14 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %13 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %17 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %17, label %18 [
    i32 0, label %18
    i32 1, label %19
  ]

18:                                               ; preds = %15, %15
  br label %20

19:                                               ; preds = %15
  br label %20

20:                                               ; preds = %19, %18
  %21 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %18 ]
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %22, ptr %23, ptr @C_L1L2_0_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %24 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %25 = add i32 %24, 1
  %26 = icmp sge i32 %25, 2
  %27 = add i32 %24, -1
  %28 = select i1 %26, i32 %27, i32 %25
  store i32 %28, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %29 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %30 = add i32 %29, 1
  %31 = icmp sge i32 %30, 2
  %32 = add i32 %29, -1
  %33 = select i1 %31, i32 %32, i32 %30
  store i32 %33, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %34 = add i64 %9, 1
  br label %8

35:                                               ; preds = %8
  %36 = add i64 %6, 1
  br label %5

37:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %38 = load i32, ptr @_anonymous0, align 4
  %39 = add i32 %38, 1
  %40 = icmp sge i32 %39, 1
  %41 = select i1 %40, i32 %38, i32 %39
  store i32 %41, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_0_0_buff_0)
  br label %42

42:                                               ; preds = %72, %37
  %43 = phi i64 [ %73, %72 ], [ 0, %37 ]
  %44 = icmp slt i64 %43, 2
  br i1 %44, label %45, label %74

45:                                               ; preds = %57, %42
  %46 = phi i64 [ %71, %57 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 8
  br i1 %47, label %48, label %72

48:                                               ; preds = %45
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %49, label %50 [
    i32 0, label %50
    i32 1, label %51
  ]

50:                                               ; preds = %48, %48
  br label %52

51:                                               ; preds = %48
  br label %52

52:                                               ; preds = %51, %50
  %53 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %51 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %50 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %54, label %55 [
    i32 0, label %55
    i32 1, label %56
  ]

55:                                               ; preds = %52, %52
  br label %57

56:                                               ; preds = %52
  br label %57

57:                                               ; preds = %56, %55
  %58 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %56 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %55 ]
  %59 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, 1
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %59, ptr %60, ptr @C_L1L2_0_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %61 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %62 = add i32 %61, 1
  %63 = icmp sge i32 %62, 2
  %64 = add i32 %61, -1
  %65 = select i1 %63, i32 %64, i32 %62
  store i32 %65, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %66 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 2
  %69 = add i32 %66, -1
  %70 = select i1 %68, i32 %69, i32 %67
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %71 = add i64 %46, 1
  br label %45

72:                                               ; preds = %45
  %73 = add i64 %43, 1
  br label %42

74:                                               ; preds = %42
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %75 = load i32, ptr @_anonymous0, align 4
  %76 = add i32 %75, 1
  %77 = icmp sge i32 %76, 1
  %78 = select i1 %77, i32 %75, i32 %76
  store i32 %78, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_0_0_buff_0)
  br label %79

79:                                               ; preds = %109, %74
  %80 = phi i64 [ %110, %109 ], [ 0, %74 ]
  %81 = icmp slt i64 %80, 2
  br i1 %81, label %82, label %111

82:                                               ; preds = %94, %79
  %83 = phi i64 [ %108, %94 ], [ 0, %79 ]
  %84 = icmp slt i64 %83, 8
  br i1 %84, label %85, label %109

85:                                               ; preds = %82
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %86 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %86, label %87 [
    i32 0, label %87
    i32 1, label %88
  ]

87:                                               ; preds = %85, %85
  br label %89

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %88, %87
  %90 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %88 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %87 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %91 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %91, label %92 [
    i32 0, label %92
    i32 1, label %93
  ]

92:                                               ; preds = %89, %89
  br label %94

93:                                               ; preds = %89
  br label %94

94:                                               ; preds = %93, %92
  %95 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %93 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %92 ]
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, 1
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %96, ptr %97, ptr @C_L1L2_0_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %98 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %99 = add i32 %98, 1
  %100 = icmp sge i32 %99, 2
  %101 = add i32 %98, -1
  %102 = select i1 %100, i32 %101, i32 %99
  store i32 %102, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %103 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %104 = add i32 %103, 1
  %105 = icmp sge i32 %104, 2
  %106 = add i32 %103, -1
  %107 = select i1 %105, i32 %106, i32 %104
  store i32 %107, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %108 = add i64 %83, 1
  br label %82

109:                                              ; preds = %82
  %110 = add i64 %80, 1
  br label %79

111:                                              ; preds = %79
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %112 = load i32, ptr @_anonymous0, align 4
  %113 = add i32 %112, 1
  %114 = icmp sge i32 %113, 1
  %115 = select i1 %114, i32 %112, i32 %113
  store i32 %115, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @C_L1L2_0_0_buff_0)
  br label %116

116:                                              ; preds = %146, %111
  %117 = phi i64 [ %147, %146 ], [ 0, %111 ]
  %118 = icmp slt i64 %117, 2
  br i1 %118, label %119, label %148

119:                                              ; preds = %131, %116
  %120 = phi i64 [ %145, %131 ], [ 0, %116 ]
  %121 = icmp slt i64 %120, 8
  br i1 %121, label %122, label %146

122:                                              ; preds = %119
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %123 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %123, label %124 [
    i32 0, label %124
    i32 1, label %125
  ]

124:                                              ; preds = %122, %122
  br label %126

125:                                              ; preds = %122
  br label %126

126:                                              ; preds = %125, %124
  %127 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %125 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @B_L2L1_0_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %124 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %128 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %128, label %129 [
    i32 0, label %129
    i32 1, label %130
  ]

129:                                              ; preds = %126, %126
  br label %131

130:                                              ; preds = %126
  br label %131

131:                                              ; preds = %130, %129
  %132 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %130 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %129 ]
  %133 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, 1
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %133, ptr %134, ptr @C_L1L2_0_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %135 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %136 = add i32 %135, 1
  %137 = icmp sge i32 %136, 2
  %138 = add i32 %135, -1
  %139 = select i1 %137, i32 %138, i32 %136
  store i32 %139, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %140 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %141 = add i32 %140, 1
  %142 = icmp sge i32 %141, 2
  %143 = add i32 %140, -1
  %144 = select i1 %142, i32 %143, i32 %141
  store i32 %144, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %145 = add i64 %120, 1
  br label %119

146:                                              ; preds = %119
  %147 = add i64 %117, 1
  br label %116

148:                                              ; preds = %116
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %149 = load i32, ptr @_anonymous0, align 4
  %150 = add i32 %149, 1
  %151 = icmp sge i32 %150, 1
  %152 = select i1 %151, i32 %149, i32 %150
  store i32 %152, ptr @_anonymous0, align 4
  %153 = add i64 %2, 1
  br label %1

154:                                              ; preds = %1
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
