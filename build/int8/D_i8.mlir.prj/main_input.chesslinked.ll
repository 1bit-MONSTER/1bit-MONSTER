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
@C_L1L2_0_0_buff_0 = external global [128 x [128 x i16]]
@C_L1L2_1_0_buff_0 = external global [128 x [128 x i16]]
@C_L1L2_2_0_buff_0 = external global [128 x [128 x i16]]
@C_L1L2_3_0_buff_0 = external global [128 x [128 x i16]]
@C_L1L2_4_0_buff_0 = external global [128 x [128 x i16]]
@C_L1L2_5_0_buff_0 = external global [128 x [128 x i16]]
@C_L1L2_6_0_buff_0 = external global [128 x [128 x i16]]
@C_L1L2_7_0_buff_0 = external global [128 x [128 x i16]]

define void @core_7_2() addrspace(0) {
  store i32 0, ptr @_anonymous7, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  br label %1

1:                                                ; preds = %65, %0
  %2 = phi i64 [ %70, %65 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %71

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_7_0_buff_0)
  br label %5

5:                                                ; preds = %51, %4
  %6 = phi i64 [ %64, %51 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 48
  br i1 %7, label %8, label %65

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
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %26, label %27 [
    i32 0, label %27
    i32 1, label %28
  ]

27:                                               ; preds = %17, %17
  br label %29

28:                                               ; preds = %17
  br label %29

29:                                               ; preds = %28, %27
  %30 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %28 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %27 ]
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  call addrspace(0) void @matmul_i8_i16(ptr %31, ptr %20, ptr @C_L1L2_7_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %37 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %37, label %38 [
    i32 0, label %38
    i32 1, label %39
  ]

38:                                               ; preds = %29, %29
  br label %40

39:                                               ; preds = %29
  br label %40

40:                                               ; preds = %39, %38
  %41 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %39 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %38 ]
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  call addrspace(0) void @matmul_i8_i16(ptr %42, ptr %20, ptr @C_L1L2_7_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %43 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %44 = add i32 %43, 1
  %45 = icmp sge i32 %44, 2
  %46 = add i32 %43, -1
  %47 = select i1 %45, i32 %46, i32 %44
  store i32 %47, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %48 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %48, label %49 [
    i32 0, label %49
    i32 1, label %50
  ]

49:                                               ; preds = %40, %40
  br label %51

50:                                               ; preds = %40
  br label %51

51:                                               ; preds = %50, %49
  %52 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %50 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %49 ]
  %53 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, 1
  call addrspace(0) void @matmul_i8_i16(ptr %53, ptr %20, ptr @C_L1L2_7_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %59 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %60 = add i32 %59, 1
  %61 = icmp sge i32 %60, 2
  %62 = add i32 %59, -1
  %63 = select i1 %61, i32 %62, i32 %60
  store i32 %63, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %64 = add i64 %6, 1
  br label %5

65:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %66 = load i32, ptr @_anonymous7, align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 1
  %69 = select i1 %68, i32 %66, i32 %67
  store i32 %69, ptr @_anonymous7, align 4
  %70 = add i64 %2, 1
  br label %1

71:                                               ; preds = %1
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

1:                                                ; preds = %65, %0
  %2 = phi i64 [ %70, %65 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %71

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_6_0_buff_0)
  br label %5

5:                                                ; preds = %51, %4
  %6 = phi i64 [ %64, %51 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 48
  br i1 %7, label %8, label %65

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
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %26, label %27 [
    i32 0, label %27
    i32 1, label %28
  ]

27:                                               ; preds = %17, %17
  br label %29

28:                                               ; preds = %17
  br label %29

29:                                               ; preds = %28, %27
  %30 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %28 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %27 ]
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  call addrspace(0) void @matmul_i8_i16(ptr %31, ptr %20, ptr @C_L1L2_6_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %37 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %37, label %38 [
    i32 0, label %38
    i32 1, label %39
  ]

38:                                               ; preds = %29, %29
  br label %40

39:                                               ; preds = %29
  br label %40

40:                                               ; preds = %39, %38
  %41 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %39 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %38 ]
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  call addrspace(0) void @matmul_i8_i16(ptr %42, ptr %20, ptr @C_L1L2_6_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %43 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %44 = add i32 %43, 1
  %45 = icmp sge i32 %44, 2
  %46 = add i32 %43, -1
  %47 = select i1 %45, i32 %46, i32 %44
  store i32 %47, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %48 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %48, label %49 [
    i32 0, label %49
    i32 1, label %50
  ]

49:                                               ; preds = %40, %40
  br label %51

50:                                               ; preds = %40
  br label %51

51:                                               ; preds = %50, %49
  %52 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %50 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %49 ]
  %53 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, 1
  call addrspace(0) void @matmul_i8_i16(ptr %53, ptr %20, ptr @C_L1L2_6_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %59 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %60 = add i32 %59, 1
  %61 = icmp sge i32 %60, 2
  %62 = add i32 %59, -1
  %63 = select i1 %61, i32 %62, i32 %60
  store i32 %63, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %64 = add i64 %6, 1
  br label %5

65:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %66 = load i32, ptr @_anonymous6, align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 1
  %69 = select i1 %68, i32 %66, i32 %67
  store i32 %69, ptr @_anonymous6, align 4
  %70 = add i64 %2, 1
  br label %1

71:                                               ; preds = %1
  ret void
}

define void @core_5_2() addrspace(0) {
  store i32 0, ptr @_anonymous5, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  br label %1

1:                                                ; preds = %65, %0
  %2 = phi i64 [ %70, %65 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %71

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_5_0_buff_0)
  br label %5

5:                                                ; preds = %51, %4
  %6 = phi i64 [ %64, %51 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 48
  br i1 %7, label %8, label %65

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
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %26, label %27 [
    i32 0, label %27
    i32 1, label %28
  ]

27:                                               ; preds = %17, %17
  br label %29

28:                                               ; preds = %17
  br label %29

29:                                               ; preds = %28, %27
  %30 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %28 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %27 ]
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  call addrspace(0) void @matmul_i8_i16(ptr %31, ptr %20, ptr @C_L1L2_5_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %37 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %37, label %38 [
    i32 0, label %38
    i32 1, label %39
  ]

38:                                               ; preds = %29, %29
  br label %40

39:                                               ; preds = %29
  br label %40

40:                                               ; preds = %39, %38
  %41 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %39 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %38 ]
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  call addrspace(0) void @matmul_i8_i16(ptr %42, ptr %20, ptr @C_L1L2_5_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %43 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %44 = add i32 %43, 1
  %45 = icmp sge i32 %44, 2
  %46 = add i32 %43, -1
  %47 = select i1 %45, i32 %46, i32 %44
  store i32 %47, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %48 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %48, label %49 [
    i32 0, label %49
    i32 1, label %50
  ]

49:                                               ; preds = %40, %40
  br label %51

50:                                               ; preds = %40
  br label %51

51:                                               ; preds = %50, %49
  %52 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %50 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %49 ]
  %53 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, 1
  call addrspace(0) void @matmul_i8_i16(ptr %53, ptr %20, ptr @C_L1L2_5_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %59 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %60 = add i32 %59, 1
  %61 = icmp sge i32 %60, 2
  %62 = add i32 %59, -1
  %63 = select i1 %61, i32 %62, i32 %60
  store i32 %63, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %64 = add i64 %6, 1
  br label %5

65:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %66 = load i32, ptr @_anonymous5, align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 1
  %69 = select i1 %68, i32 %66, i32 %67
  store i32 %69, ptr @_anonymous5, align 4
  %70 = add i64 %2, 1
  br label %1

71:                                               ; preds = %1
  ret void
}

define void @core_4_2() addrspace(0) {
  store i32 0, ptr @_anonymous4, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  br label %1

1:                                                ; preds = %65, %0
  %2 = phi i64 [ %70, %65 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %71

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_4_0_buff_0)
  br label %5

5:                                                ; preds = %51, %4
  %6 = phi i64 [ %64, %51 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 48
  br i1 %7, label %8, label %65

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
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %26, label %27 [
    i32 0, label %27
    i32 1, label %28
  ]

27:                                               ; preds = %17, %17
  br label %29

28:                                               ; preds = %17
  br label %29

29:                                               ; preds = %28, %27
  %30 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %28 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %27 ]
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  call addrspace(0) void @matmul_i8_i16(ptr %31, ptr %20, ptr @C_L1L2_4_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %37 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %37, label %38 [
    i32 0, label %38
    i32 1, label %39
  ]

38:                                               ; preds = %29, %29
  br label %40

39:                                               ; preds = %29
  br label %40

40:                                               ; preds = %39, %38
  %41 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %39 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %38 ]
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  call addrspace(0) void @matmul_i8_i16(ptr %42, ptr %20, ptr @C_L1L2_4_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %43 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %44 = add i32 %43, 1
  %45 = icmp sge i32 %44, 2
  %46 = add i32 %43, -1
  %47 = select i1 %45, i32 %46, i32 %44
  store i32 %47, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %48 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %48, label %49 [
    i32 0, label %49
    i32 1, label %50
  ]

49:                                               ; preds = %40, %40
  br label %51

50:                                               ; preds = %40
  br label %51

51:                                               ; preds = %50, %49
  %52 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %50 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %49 ]
  %53 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, 1
  call addrspace(0) void @matmul_i8_i16(ptr %53, ptr %20, ptr @C_L1L2_4_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %59 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %60 = add i32 %59, 1
  %61 = icmp sge i32 %60, 2
  %62 = add i32 %59, -1
  %63 = select i1 %61, i32 %62, i32 %60
  store i32 %63, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %64 = add i64 %6, 1
  br label %5

65:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %66 = load i32, ptr @_anonymous4, align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 1
  %69 = select i1 %68, i32 %66, i32 %67
  store i32 %69, ptr @_anonymous4, align 4
  %70 = add i64 %2, 1
  br label %1

71:                                               ; preds = %1
  ret void
}

define void @core_3_2() addrspace(0) {
  store i32 0, ptr @_anonymous3, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  br label %1

1:                                                ; preds = %65, %0
  %2 = phi i64 [ %70, %65 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %71

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_3_0_buff_0)
  br label %5

5:                                                ; preds = %51, %4
  %6 = phi i64 [ %64, %51 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 48
  br i1 %7, label %8, label %65

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
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %26, label %27 [
    i32 0, label %27
    i32 1, label %28
  ]

27:                                               ; preds = %17, %17
  br label %29

28:                                               ; preds = %17
  br label %29

29:                                               ; preds = %28, %27
  %30 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %28 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %27 ]
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  call addrspace(0) void @matmul_i8_i16(ptr %31, ptr %20, ptr @C_L1L2_3_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %37 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %37, label %38 [
    i32 0, label %38
    i32 1, label %39
  ]

38:                                               ; preds = %29, %29
  br label %40

39:                                               ; preds = %29
  br label %40

40:                                               ; preds = %39, %38
  %41 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %39 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %38 ]
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  call addrspace(0) void @matmul_i8_i16(ptr %42, ptr %20, ptr @C_L1L2_3_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %43 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %44 = add i32 %43, 1
  %45 = icmp sge i32 %44, 2
  %46 = add i32 %43, -1
  %47 = select i1 %45, i32 %46, i32 %44
  store i32 %47, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %48 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %48, label %49 [
    i32 0, label %49
    i32 1, label %50
  ]

49:                                               ; preds = %40, %40
  br label %51

50:                                               ; preds = %40
  br label %51

51:                                               ; preds = %50, %49
  %52 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %50 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %49 ]
  %53 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, 1
  call addrspace(0) void @matmul_i8_i16(ptr %53, ptr %20, ptr @C_L1L2_3_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %59 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %60 = add i32 %59, 1
  %61 = icmp sge i32 %60, 2
  %62 = add i32 %59, -1
  %63 = select i1 %61, i32 %62, i32 %60
  store i32 %63, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %64 = add i64 %6, 1
  br label %5

65:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %66 = load i32, ptr @_anonymous3, align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 1
  %69 = select i1 %68, i32 %66, i32 %67
  store i32 %69, ptr @_anonymous3, align 4
  %70 = add i64 %2, 1
  br label %1

71:                                               ; preds = %1
  ret void
}

define void @core_2_2() addrspace(0) {
  store i32 0, ptr @_anonymous2, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  br label %1

1:                                                ; preds = %65, %0
  %2 = phi i64 [ %70, %65 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %71

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_2_0_buff_0)
  br label %5

5:                                                ; preds = %51, %4
  %6 = phi i64 [ %64, %51 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 48
  br i1 %7, label %8, label %65

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
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %26, label %27 [
    i32 0, label %27
    i32 1, label %28
  ]

27:                                               ; preds = %17, %17
  br label %29

28:                                               ; preds = %17
  br label %29

29:                                               ; preds = %28, %27
  %30 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %28 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %27 ]
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  call addrspace(0) void @matmul_i8_i16(ptr %31, ptr %20, ptr @C_L1L2_2_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %37 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %37, label %38 [
    i32 0, label %38
    i32 1, label %39
  ]

38:                                               ; preds = %29, %29
  br label %40

39:                                               ; preds = %29
  br label %40

40:                                               ; preds = %39, %38
  %41 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %39 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %38 ]
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  call addrspace(0) void @matmul_i8_i16(ptr %42, ptr %20, ptr @C_L1L2_2_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %43 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %44 = add i32 %43, 1
  %45 = icmp sge i32 %44, 2
  %46 = add i32 %43, -1
  %47 = select i1 %45, i32 %46, i32 %44
  store i32 %47, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %48 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %48, label %49 [
    i32 0, label %49
    i32 1, label %50
  ]

49:                                               ; preds = %40, %40
  br label %51

50:                                               ; preds = %40
  br label %51

51:                                               ; preds = %50, %49
  %52 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %50 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %49 ]
  %53 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, 1
  call addrspace(0) void @matmul_i8_i16(ptr %53, ptr %20, ptr @C_L1L2_2_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %59 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %60 = add i32 %59, 1
  %61 = icmp sge i32 %60, 2
  %62 = add i32 %59, -1
  %63 = select i1 %61, i32 %62, i32 %60
  store i32 %63, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %64 = add i64 %6, 1
  br label %5

65:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %66 = load i32, ptr @_anonymous2, align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 1
  %69 = select i1 %68, i32 %66, i32 %67
  store i32 %69, ptr @_anonymous2, align 4
  %70 = add i64 %2, 1
  br label %1

71:                                               ; preds = %1
  ret void
}

define void @core_1_2() addrspace(0) {
  store i32 0, ptr @_anonymous1, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  br label %1

1:                                                ; preds = %65, %0
  %2 = phi i64 [ %70, %65 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %71

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_1_0_buff_0)
  br label %5

5:                                                ; preds = %51, %4
  %6 = phi i64 [ %64, %51 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 48
  br i1 %7, label %8, label %65

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
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %26, label %27 [
    i32 0, label %27
    i32 1, label %28
  ]

27:                                               ; preds = %17, %17
  br label %29

28:                                               ; preds = %17
  br label %29

29:                                               ; preds = %28, %27
  %30 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %28 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %27 ]
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  call addrspace(0) void @matmul_i8_i16(ptr %31, ptr %20, ptr @C_L1L2_1_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %37 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %37, label %38 [
    i32 0, label %38
    i32 1, label %39
  ]

38:                                               ; preds = %29, %29
  br label %40

39:                                               ; preds = %29
  br label %40

40:                                               ; preds = %39, %38
  %41 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %39 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %38 ]
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  call addrspace(0) void @matmul_i8_i16(ptr %42, ptr %20, ptr @C_L1L2_1_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %43 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %44 = add i32 %43, 1
  %45 = icmp sge i32 %44, 2
  %46 = add i32 %43, -1
  %47 = select i1 %45, i32 %46, i32 %44
  store i32 %47, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %48 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %48, label %49 [
    i32 0, label %49
    i32 1, label %50
  ]

49:                                               ; preds = %40, %40
  br label %51

50:                                               ; preds = %40
  br label %51

51:                                               ; preds = %50, %49
  %52 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %50 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %49 ]
  %53 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, 1
  call addrspace(0) void @matmul_i8_i16(ptr %53, ptr %20, ptr @C_L1L2_1_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %59 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %60 = add i32 %59, 1
  %61 = icmp sge i32 %60, 2
  %62 = add i32 %59, -1
  %63 = select i1 %61, i32 %62, i32 %60
  store i32 %63, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %64 = add i64 %6, 1
  br label %5

65:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %66 = load i32, ptr @_anonymous1, align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 1
  %69 = select i1 %68, i32 %66, i32 %67
  store i32 %69, ptr @_anonymous1, align 4
  %70 = add i64 %2, 1
  br label %1

71:                                               ; preds = %1
  ret void
}

define void @core_0_2() addrspace(0) {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  br label %1

1:                                                ; preds = %65, %0
  %2 = phi i64 [ %70, %65 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %71

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @C_L1L2_0_0_buff_0)
  br label %5

5:                                                ; preds = %51, %4
  %6 = phi i64 [ %64, %51 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 48
  br i1 %7, label %8, label %65

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
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %26 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %26, label %27 [
    i32 0, label %27
    i32 1, label %28
  ]

27:                                               ; preds = %17, %17
  br label %29

28:                                               ; preds = %17
  br label %29

29:                                               ; preds = %28, %27
  %30 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %28 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %27 ]
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 1
  call addrspace(0) void @matmul_i8_i16(ptr %31, ptr %20, ptr @C_L1L2_0_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %37 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %37, label %38 [
    i32 0, label %38
    i32 1, label %39
  ]

38:                                               ; preds = %29, %29
  br label %40

39:                                               ; preds = %29
  br label %40

40:                                               ; preds = %39, %38
  %41 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %39 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %38 ]
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  call addrspace(0) void @matmul_i8_i16(ptr %42, ptr %20, ptr @C_L1L2_0_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %43 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %44 = add i32 %43, 1
  %45 = icmp sge i32 %44, 2
  %46 = add i32 %43, -1
  %47 = select i1 %45, i32 %46, i32 %44
  store i32 %47, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %48 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %48, label %49 [
    i32 0, label %49
    i32 1, label %50
  ]

49:                                               ; preds = %40, %40
  br label %51

50:                                               ; preds = %40
  br label %51

51:                                               ; preds = %50, %49
  %52 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %50 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %49 ]
  %53 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, 1
  call addrspace(0) void @matmul_i8_i16(ptr %53, ptr %20, ptr @C_L1L2_0_0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %59 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %60 = add i32 %59, 1
  %61 = icmp sge i32 %60, 2
  %62 = add i32 %59, -1
  %63 = select i1 %61, i32 %62, i32 %60
  store i32 %63, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %64 = add i64 %6, 1
  br label %5

65:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %66 = load i32, ptr @_anonymous0, align 4
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 1
  %69 = select i1 %68, i32 %66, i32 %67
  store i32 %69, ptr @_anonymous0, align 4
  %70 = add i64 %2, 1
  br label %1

71:                                               ; preds = %1
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
