; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@_anonymous7 = external global [3 x i32]
@_anonymous6 = external global [3 x i32]
@_anonymous5 = external global [3 x i32]
@_anonymous4 = external global [3 x i32]
@_anonymous3 = external global [3 x i32]
@_anonymous2 = external global [3 x i32]
@_anonymous1 = external global [3 x i32]
@_anonymous0 = external global [3 x i32]
@A_L3L2_cons_buff_1 = external global [32 x [64 x i8]]
@A_L3L2_cons_buff_0 = external global [32 x [64 x i8]]
@A_L2L1_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_buff_0 = external global [32 x [64 x i8]]
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
@BS0_cons_buff_1 = external global [64 x [128 x i8]]
@BS0_cons_buff_0 = external global [64 x [128 x i8]]
@BC0_cons_buff_1 = external global [64 x [128 x i8]]
@BC0_cons_buff_0 = external global [64 x [128 x i8]]
@CC0_buff_0 = external global [128 x [128 x i16]]
@CC0_cons_buff_0 = external global [128 x [128 x i16]]
@BS1_cons_buff_1 = external global [64 x [128 x i8]]
@BS1_cons_buff_0 = external global [64 x [128 x i8]]
@BC1_cons_buff_1 = external global [64 x [128 x i8]]
@BC1_cons_buff_0 = external global [64 x [128 x i8]]
@CC1_buff_0 = external global [128 x [128 x i16]]
@CC1_cons_buff_0 = external global [128 x [128 x i16]]
@BS2_cons_buff_1 = external global [64 x [128 x i8]]
@BS2_cons_buff_0 = external global [64 x [128 x i8]]
@BC2_cons_buff_1 = external global [64 x [128 x i8]]
@BC2_cons_buff_0 = external global [64 x [128 x i8]]
@CC2_buff_0 = external global [128 x [128 x i16]]
@CC2_cons_buff_0 = external global [128 x [128 x i16]]
@BS3_cons_buff_1 = external global [64 x [128 x i8]]
@BS3_cons_buff_0 = external global [64 x [128 x i8]]
@BC3_cons_buff_1 = external global [64 x [128 x i8]]
@BC3_cons_buff_0 = external global [64 x [128 x i8]]
@CC3_buff_0 = external global [128 x [128 x i16]]
@CC3_cons_buff_0 = external global [128 x [128 x i16]]
@BS4_cons_buff_1 = external global [64 x [128 x i8]]
@BS4_cons_buff_0 = external global [64 x [128 x i8]]
@BC4_cons_buff_1 = external global [64 x [128 x i8]]
@BC4_cons_buff_0 = external global [64 x [128 x i8]]
@CC4_buff_0 = external global [128 x [128 x i16]]
@CC4_cons_buff_0 = external global [128 x [128 x i16]]
@BS5_cons_buff_1 = external global [64 x [128 x i8]]
@BS5_cons_buff_0 = external global [64 x [128 x i8]]
@BC5_cons_buff_1 = external global [64 x [128 x i8]]
@BC5_cons_buff_0 = external global [64 x [128 x i8]]
@CC5_buff_0 = external global [128 x [128 x i16]]
@CC5_cons_buff_0 = external global [128 x [128 x i16]]
@BS6_cons_buff_1 = external global [64 x [128 x i8]]
@BS6_cons_buff_0 = external global [64 x [128 x i8]]
@BC6_cons_buff_1 = external global [64 x [128 x i8]]
@BC6_cons_buff_0 = external global [64 x [128 x i8]]
@CC6_buff_0 = external global [128 x [128 x i16]]
@CC6_cons_buff_0 = external global [128 x [128 x i16]]
@BS7_cons_buff_1 = external global [64 x [128 x i8]]
@BS7_cons_buff_0 = external global [64 x [128 x i8]]
@BC7_cons_buff_1 = external global [64 x [128 x i8]]
@BC7_cons_buff_0 = external global [64 x [128 x i8]]
@CC7_buff_0 = external global [128 x [128 x i16]]
@CC7_cons_buff_0 = external global [128 x [128 x i16]]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.event(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.set.ctrl.reg(i32, i32)

declare void @zero_scalar_i16(ptr)

declare void @matmul_scalar_i8_i16(ptr, ptr, ptr)

define void @core_7_2() {
  store i32 0, ptr @_anonymous7, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous7, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous7, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_scalar_i16(ptr @CC7_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

8:                                                ; preds = %5
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous7, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %10, %11
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC7_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC7_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous7, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %19, %20
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_7_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_7_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC7_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous7, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds nuw (i8, ptr @_anonymous7, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous7, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @_anonymous7, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 53, i32 1)
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

define void @core_6_2() {
  store i32 0, ptr @_anonymous6, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_scalar_i16(ptr @CC6_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

8:                                                ; preds = %5
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %10, %11
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC6_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC6_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %19, %20
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_6_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_6_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC6_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 53, i32 1)
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

define void @core_5_2() {
  store i32 0, ptr @_anonymous5, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_scalar_i16(ptr @CC5_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

8:                                                ; preds = %5
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %10, %11
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC5_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC5_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %19, %20
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_5_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_5_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC5_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 53, i32 1)
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

define void @core_4_2() {
  store i32 0, ptr @_anonymous4, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_scalar_i16(ptr @CC4_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

8:                                                ; preds = %5
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %10, %11
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC4_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC4_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %19, %20
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_4_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_4_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC4_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 53, i32 1)
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

define void @core_3_2() {
  store i32 0, ptr @_anonymous3, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_scalar_i16(ptr @CC3_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

8:                                                ; preds = %5
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %10, %11
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC3_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC3_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %19, %20
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_3_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_3_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC3_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 53, i32 1)
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

define void @core_2_2() {
  store i32 0, ptr @_anonymous2, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_scalar_i16(ptr @CC2_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

8:                                                ; preds = %5
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %10, %11
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC2_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC2_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %19, %20
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_2_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_2_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC2_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 53, i32 1)
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

define void @core_1_2() {
  store i32 0, ptr @_anonymous1, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_scalar_i16(ptr @CC1_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

8:                                                ; preds = %5
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %10, %11
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC1_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC1_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %19, %20
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_1_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_1_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC1_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 53, i32 1)
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

define void @core_0_2() {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %43, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %44

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_scalar_i16(ptr @CC0_buff_0)
  br label %5

5:                                                ; preds = %31, %4
  %6 = phi i64 [ %37, %31 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %38

8:                                                ; preds = %5
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %11
  ]

10:                                               ; preds = %8, %8
  br label %12

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %10, %11
  %13 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC0_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %11 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @BC0_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %10 ]
  br label %14

14:                                               ; preds = %21, %12
  %15 = phi i64 [ %30, %21 ], [ 0, %12 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %31

17:                                               ; preds = %14
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %18, label %19 [
    i32 0, label %19
    i32 1, label %20
  ]

19:                                               ; preds = %17, %17
  br label %21

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %19, %20
  %22 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %20 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @A_L2L1_0_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %19 ]
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  call void @matmul_scalar_i8_i16(ptr %23, ptr %24, ptr @CC0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  %30 = add i64 %15, 1
  br label %14

31:                                               ; preds = %14
  call void @llvm.aie2p.release(i32 50, i32 1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  %37 = add i64 %6, 1
  br label %5

38:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 53, i32 1)
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

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
