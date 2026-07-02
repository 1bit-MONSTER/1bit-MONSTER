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
@A_L2L1_7_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_7_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_6_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_6_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_5_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_5_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_4_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_4_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_3_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_3_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_2_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_2_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_1_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_1_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_0_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_0_cons_buff_0 = external global [32 x [64 x i8]]
@BC0_cons_buff_1 = external global [64 x [128 x i8]]
@BC0_cons_buff_0 = external global [64 x [128 x i8]]
@CC0_buff_0 = external global [128 x [128 x i16]]
@BC1_cons_buff_1 = external global [64 x [128 x i8]]
@BC1_cons_buff_0 = external global [64 x [128 x i8]]
@CC1_buff_0 = external global [128 x [128 x i16]]
@BC2_cons_buff_1 = external global [64 x [128 x i8]]
@BC2_cons_buff_0 = external global [64 x [128 x i8]]
@CC2_buff_0 = external global [128 x [128 x i16]]
@BC3_cons_buff_1 = external global [64 x [128 x i8]]
@BC3_cons_buff_0 = external global [64 x [128 x i8]]
@CC3_buff_0 = external global [128 x [128 x i16]]
@BC4_cons_buff_1 = external global [64 x [128 x i8]]
@BC4_cons_buff_0 = external global [64 x [128 x i8]]
@CC4_buff_0 = external global [128 x [128 x i16]]
@BC5_cons_buff_1 = external global [64 x [128 x i8]]
@BC5_cons_buff_0 = external global [64 x [128 x i8]]
@CC5_buff_0 = external global [128 x [128 x i16]]
@BC6_cons_buff_1 = external global [64 x [128 x i8]]
@BC6_cons_buff_0 = external global [64 x [128 x i8]]
@CC6_buff_0 = external global [128 x [128 x i16]]
@BC7_cons_buff_1 = external global [64 x [128 x i8]]
@BC7_cons_buff_0 = external global [64 x [128 x i8]]
@CC7_buff_0 = external global [128 x [128 x i16]]

define void @core_7_2() addrspace(0) {
  store i32 0, ptr @_anonymous7, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC7_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

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
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC7_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC7_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %20, %19
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC7_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %39 = load i32, ptr @_anonymous7, align 4
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 1
  %42 = select i1 %41, i32 %39, i32 %40
  store i32 %42, ptr @_anonymous7, align 4
  %43 = add i64 %2, 1
  br label %1

44:                                               ; preds = %1
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

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC6_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

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
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC6_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC6_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %20, %19
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC6_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %39 = load i32, ptr @_anonymous6, align 4
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 1
  %42 = select i1 %41, i32 %39, i32 %40
  store i32 %42, ptr @_anonymous6, align 4
  %43 = add i64 %2, 1
  br label %1

44:                                               ; preds = %1
  ret void
}

define void @core_5_2() addrspace(0) {
  store i32 0, ptr @_anonymous5, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC5_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

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
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC5_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC5_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %20, %19
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC5_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %39 = load i32, ptr @_anonymous5, align 4
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 1
  %42 = select i1 %41, i32 %39, i32 %40
  store i32 %42, ptr @_anonymous5, align 4
  %43 = add i64 %2, 1
  br label %1

44:                                               ; preds = %1
  ret void
}

define void @core_4_2() addrspace(0) {
  store i32 0, ptr @_anonymous4, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC4_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

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
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC4_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC4_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %20, %19
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC4_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %39 = load i32, ptr @_anonymous4, align 4
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 1
  %42 = select i1 %41, i32 %39, i32 %40
  store i32 %42, ptr @_anonymous4, align 4
  %43 = add i64 %2, 1
  br label %1

44:                                               ; preds = %1
  ret void
}

define void @core_3_2() addrspace(0) {
  store i32 0, ptr @_anonymous3, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC3_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

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
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC3_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC3_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %20, %19
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC3_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %39 = load i32, ptr @_anonymous3, align 4
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 1
  %42 = select i1 %41, i32 %39, i32 %40
  store i32 %42, ptr @_anonymous3, align 4
  %43 = add i64 %2, 1
  br label %1

44:                                               ; preds = %1
  ret void
}

define void @core_2_2() addrspace(0) {
  store i32 0, ptr @_anonymous2, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC2_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

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
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC2_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC2_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %20, %19
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %39 = load i32, ptr @_anonymous2, align 4
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 1
  %42 = select i1 %41, i32 %39, i32 %40
  store i32 %42, ptr @_anonymous2, align 4
  %43 = add i64 %2, 1
  br label %1

44:                                               ; preds = %1
  ret void
}

define void @core_1_2() addrspace(0) {
  store i32 0, ptr @_anonymous1, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC1_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

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
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %20, %19
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC1_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %39 = load i32, ptr @_anonymous1, align 4
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 1
  %42 = select i1 %41, i32 %39, i32 %40
  store i32 %42, ptr @_anonymous1, align 4
  %43 = add i64 %2, 1
  br label %1

44:                                               ; preds = %1
  ret void
}

define void @core_0_2() addrspace(0) {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC0_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

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
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC0_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC0_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %20, %19
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call addrspace(0) void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %39 = load i32, ptr @_anonymous0, align 4
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 1
  %42 = select i1 %41, i32 %39, i32 %40
  store i32 %42, ptr @_anonymous0, align 4
  %43 = add i64 %2, 1
  br label %1

44:                                               ; preds = %1
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
