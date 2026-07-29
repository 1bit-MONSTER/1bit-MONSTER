; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@_anonymous3 = external global [3 x i32]
@_anonymous2 = external global [3 x i32]
@_anonymous1 = external global [3 x i32]
@_anonymous0 = external global [3 x i32]
@C_C0_cons_buff_1 = external global [32 x [128 x i32]]
@C_C0_cons_buff_0 = external global [32 x [128 x i32]]
@A_C_3_cons_buff_1 = external global [32 x [64 x i8]]
@A_C_3_cons_buff_0 = external global [32 x [64 x i8]]
@A_C_2_cons_buff_1 = external global [32 x [64 x i8]]
@A_C_2_cons_buff_0 = external global [32 x [64 x i8]]
@A_C_1_cons_buff_1 = external global [32 x [64 x i8]]
@A_C_1_cons_buff_0 = external global [32 x [64 x i8]]
@A_C_0_cons_buff_1 = external global [32 x [64 x i8]]
@A_C_0_cons_buff_0 = external global [32 x [64 x i8]]
@C_C1_cons_buff_1 = external global [32 x [128 x i32]]
@C_C1_cons_buff_0 = external global [32 x [128 x i32]]
@B_C0_cons_buff_1 = external global [64 x [128 x i8]]
@B_C0_cons_buff_0 = external global [64 x [128 x i8]]
@C_C0_buff_1 = external global [32 x [128 x i32]]
@C_C0_buff_0 = external global [32 x [128 x i32]]
@C_C2_cons_buff_1 = external global [32 x [128 x i32]]
@C_C2_cons_buff_0 = external global [32 x [128 x i32]]
@C_C3_cons_buff_1 = external global [32 x [128 x i32]]
@C_C3_cons_buff_0 = external global [32 x [128 x i32]]
@B_C1_cons_buff_1 = external global [64 x [128 x i8]]
@B_C1_cons_buff_0 = external global [64 x [128 x i8]]
@C_C1_buff_1 = external global [32 x [128 x i32]]
@C_C1_buff_0 = external global [32 x [128 x i32]]
@B_S0_cons_buff_1 = external global [64 x [128 x i8]]
@B_S0_cons_buff_0 = external global [64 x [128 x i8]]
@B_S1_cons_buff_1 = external global [64 x [128 x i8]]
@B_S1_cons_buff_0 = external global [64 x [128 x i8]]
@B_C2_cons_buff_1 = external global [64 x [128 x i8]]
@B_C2_cons_buff_0 = external global [64 x [128 x i8]]
@C_C2_buff_1 = external global [32 x [128 x i32]]
@C_C2_buff_0 = external global [32 x [128 x i32]]
@B_S2_cons_buff_1 = external global [64 x [128 x i8]]
@B_S2_cons_buff_0 = external global [64 x [128 x i8]]
@B_S3_cons_buff_1 = external global [64 x [128 x i8]]
@B_S3_cons_buff_0 = external global [64 x [128 x i8]]
@B_C3_cons_buff_1 = external global [64 x [128 x i8]]
@B_C3_cons_buff_0 = external global [64 x [128 x i8]]
@C_C3_buff_1 = external global [32 x [128 x i32]]
@C_C3_buff_0 = external global [32 x [128 x i32]]
@A_S_cons_buff_1 = external global [32 x [64 x i8]]
@A_S_cons_buff_0 = external global [32 x [64 x i8]]

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

declare void @zero_i32(ptr)

declare void @matmul_i8_i32(ptr, ptr, ptr)

define void @core_3_2() {
  store i32 0, ptr @_anonymous3
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4)
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8)
  br label %1

1:                                                ; preds = %74, %0
  %2 = phi i64 [ %75, %74 ], [ 0, %0 ]
  %3 = phi i32 [ %9, %74 ], [ 0, %0 ]
  %4 = phi i32 [ %10, %74 ], [ 0, %0 ]
  %5 = phi i32 [ %11, %74 ], [ 0, %0 ]
  %6 = icmp slt i64 %2, 4294967295
  br i1 %6, label %7, label %76

7:                                                ; preds = %66, %1
  %8 = phi i64 [ %73, %66 ], [ 0, %1 ]
  %9 = phi i32 [ %67, %66 ], [ %3, %1 ]
  %10 = phi i32 [ %27, %66 ], [ %4, %1 ]
  %11 = phi i32 [ %28, %66 ], [ %5, %1 ]
  %12 = icmp slt i64 %8, 4
  br i1 %12, label %13, label %74

13:                                               ; preds = %7
  %14 = sub i32 1, %9
  %15 = icmp sgt i32 %14, 0
  %16 = select i1 %15, i32 %14, i32 0
  %17 = sub i32 0, %16
  call void @llvm.aie2p.acquire(i32 52, i32 %17)
  %18 = add i32 %9, %16
  %19 = load i32, ptr @_anonymous3
  %20 = sext i32 %19 to i64
  switch i64 %20, label %21 [
    i64 0, label %77
    i64 1, label %79
  ]

21:                                               ; preds = %77, %79, %13
  %22 = phi ptr [ %80, %79 ], [ %78, %77 ], [ @C_C3_buff_0, %13 ]
  %23 = getelementptr [32 x [128 x i32]], ptr %22, i32 0, i32 0, i32 0
  br label %24

24:                                               ; preds = %21
  call void @zero_i32(ptr %23)
  br label %25

25:                                               ; preds = %52, %24
  %26 = phi i64 [ %65, %52 ], [ 0, %24 ]
  %27 = phi i32 [ %53, %52 ], [ %10, %24 ]
  %28 = phi i32 [ %59, %52 ], [ %11, %24 ]
  %29 = icmp slt i64 %26, 32
  br i1 %29, label %30, label %66

30:                                               ; preds = %25
  %31 = sub i32 1, %27
  %32 = icmp sgt i32 %31, 0
  %33 = select i1 %32, i32 %31, i32 0
  %34 = sub i32 0, %33
  call void @llvm.aie2p.acquire(i32 49, i32 %34)
  %35 = add i32 %27, %33
  %36 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4)
  %37 = sext i32 %36 to i64
  switch i64 %37, label %38 [
    i64 0, label %81
    i64 1, label %83
  ]

38:                                               ; preds = %81, %83, %30
  %39 = phi ptr [ %84, %83 ], [ %82, %81 ], [ @A_C_3_cons_buff_0, %30 ]
  %40 = getelementptr [32 x [64 x i8]], ptr %39, i32 0, i32 0, i32 0
  br label %41

41:                                               ; preds = %38
  %42 = sub i32 1, %28
  %43 = icmp sgt i32 %42, 0
  %44 = select i1 %43, i32 %42, i32 0
  %45 = sub i32 0, %44
  call void @llvm.aie2p.acquire(i32 51, i32 %45)
  %46 = add i32 %28, %44
  %47 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8)
  %48 = sext i32 %47 to i64
  switch i64 %48, label %49 [
    i64 0, label %85
    i64 1, label %87
  ]

49:                                               ; preds = %85, %87, %41
  %50 = phi ptr [ %88, %87 ], [ %86, %85 ], [ @B_C3_cons_buff_0, %41 ]
  %51 = getelementptr [64 x [128 x i8]], ptr %50, i32 0, i32 0, i32 0
  br label %52

52:                                               ; preds = %49
  call void @matmul_i8_i32(ptr %40, ptr %51, ptr %23)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %53 = sub i32 %35, 1
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4)
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %59 = sub i32 %46, 1
  %60 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8)
  %61 = add i32 %60, 1
  %62 = icmp sge i32 %61, 2
  %63 = add i32 %60, -1
  %64 = select i1 %62, i32 %63, i32 %61
  store i32 %64, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8)
  %65 = add i64 %26, 1
  br label %25

66:                                               ; preds = %25
  call void @llvm.aie2p.release(i32 53, i32 1)
  %67 = sub i32 %18, 1
  %68 = load i32, ptr @_anonymous3
  %69 = add i32 %68, 1
  %70 = icmp sge i32 %69, 2
  %71 = add i32 %68, -1
  %72 = select i1 %70, i32 %71, i32 %69
  store i32 %72, ptr @_anonymous3
  %73 = add i64 %8, 1
  br label %7

74:                                               ; preds = %7
  %75 = add i64 %2, 1
  br label %1

76:                                               ; preds = %1
  ret void

77:                                               ; preds = %13
  %78 = phi ptr [ @C_C3_buff_0, %13 ]
  br label %21

79:                                               ; preds = %13
  %80 = phi ptr [ @C_C3_buff_1, %13 ]
  br label %21

81:                                               ; preds = %30
  %82 = phi ptr [ @A_C_3_cons_buff_0, %30 ]
  br label %38

83:                                               ; preds = %30
  %84 = phi ptr [ @A_C_3_cons_buff_1, %30 ]
  br label %38

85:                                               ; preds = %41
  %86 = phi ptr [ @B_C3_cons_buff_0, %41 ]
  br label %49

87:                                               ; preds = %41
  %88 = phi ptr [ @B_C3_cons_buff_1, %41 ]
  br label %49
}

define void @core_2_2() {
  store i32 0, ptr @_anonymous2
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4)
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8)
  br label %1

1:                                                ; preds = %74, %0
  %2 = phi i64 [ %75, %74 ], [ 0, %0 ]
  %3 = phi i32 [ %9, %74 ], [ 0, %0 ]
  %4 = phi i32 [ %10, %74 ], [ 0, %0 ]
  %5 = phi i32 [ %11, %74 ], [ 0, %0 ]
  %6 = icmp slt i64 %2, 4294967295
  br i1 %6, label %7, label %76

7:                                                ; preds = %66, %1
  %8 = phi i64 [ %73, %66 ], [ 0, %1 ]
  %9 = phi i32 [ %67, %66 ], [ %3, %1 ]
  %10 = phi i32 [ %27, %66 ], [ %4, %1 ]
  %11 = phi i32 [ %28, %66 ], [ %5, %1 ]
  %12 = icmp slt i64 %8, 4
  br i1 %12, label %13, label %74

13:                                               ; preds = %7
  %14 = sub i32 1, %9
  %15 = icmp sgt i32 %14, 0
  %16 = select i1 %15, i32 %14, i32 0
  %17 = sub i32 0, %16
  call void @llvm.aie2p.acquire(i32 52, i32 %17)
  %18 = add i32 %9, %16
  %19 = load i32, ptr @_anonymous2
  %20 = sext i32 %19 to i64
  switch i64 %20, label %21 [
    i64 0, label %77
    i64 1, label %79
  ]

21:                                               ; preds = %77, %79, %13
  %22 = phi ptr [ %80, %79 ], [ %78, %77 ], [ @C_C2_buff_0, %13 ]
  %23 = getelementptr [32 x [128 x i32]], ptr %22, i32 0, i32 0, i32 0
  br label %24

24:                                               ; preds = %21
  call void @zero_i32(ptr %23)
  br label %25

25:                                               ; preds = %52, %24
  %26 = phi i64 [ %65, %52 ], [ 0, %24 ]
  %27 = phi i32 [ %53, %52 ], [ %10, %24 ]
  %28 = phi i32 [ %59, %52 ], [ %11, %24 ]
  %29 = icmp slt i64 %26, 32
  br i1 %29, label %30, label %66

30:                                               ; preds = %25
  %31 = sub i32 1, %27
  %32 = icmp sgt i32 %31, 0
  %33 = select i1 %32, i32 %31, i32 0
  %34 = sub i32 0, %33
  call void @llvm.aie2p.acquire(i32 49, i32 %34)
  %35 = add i32 %27, %33
  %36 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4)
  %37 = sext i32 %36 to i64
  switch i64 %37, label %38 [
    i64 0, label %81
    i64 1, label %83
  ]

38:                                               ; preds = %81, %83, %30
  %39 = phi ptr [ %84, %83 ], [ %82, %81 ], [ @A_C_2_cons_buff_0, %30 ]
  %40 = getelementptr [32 x [64 x i8]], ptr %39, i32 0, i32 0, i32 0
  br label %41

41:                                               ; preds = %38
  %42 = sub i32 1, %28
  %43 = icmp sgt i32 %42, 0
  %44 = select i1 %43, i32 %42, i32 0
  %45 = sub i32 0, %44
  call void @llvm.aie2p.acquire(i32 51, i32 %45)
  %46 = add i32 %28, %44
  %47 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8)
  %48 = sext i32 %47 to i64
  switch i64 %48, label %49 [
    i64 0, label %85
    i64 1, label %87
  ]

49:                                               ; preds = %85, %87, %41
  %50 = phi ptr [ %88, %87 ], [ %86, %85 ], [ @B_C2_cons_buff_0, %41 ]
  %51 = getelementptr [64 x [128 x i8]], ptr %50, i32 0, i32 0, i32 0
  br label %52

52:                                               ; preds = %49
  call void @matmul_i8_i32(ptr %40, ptr %51, ptr %23)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %53 = sub i32 %35, 1
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4)
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %59 = sub i32 %46, 1
  %60 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8)
  %61 = add i32 %60, 1
  %62 = icmp sge i32 %61, 2
  %63 = add i32 %60, -1
  %64 = select i1 %62, i32 %63, i32 %61
  store i32 %64, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8)
  %65 = add i64 %26, 1
  br label %25

66:                                               ; preds = %25
  call void @llvm.aie2p.release(i32 53, i32 1)
  %67 = sub i32 %18, 1
  %68 = load i32, ptr @_anonymous2
  %69 = add i32 %68, 1
  %70 = icmp sge i32 %69, 2
  %71 = add i32 %68, -1
  %72 = select i1 %70, i32 %71, i32 %69
  store i32 %72, ptr @_anonymous2
  %73 = add i64 %8, 1
  br label %7

74:                                               ; preds = %7
  %75 = add i64 %2, 1
  br label %1

76:                                               ; preds = %1
  ret void

77:                                               ; preds = %13
  %78 = phi ptr [ @C_C2_buff_0, %13 ]
  br label %21

79:                                               ; preds = %13
  %80 = phi ptr [ @C_C2_buff_1, %13 ]
  br label %21

81:                                               ; preds = %30
  %82 = phi ptr [ @A_C_2_cons_buff_0, %30 ]
  br label %38

83:                                               ; preds = %30
  %84 = phi ptr [ @A_C_2_cons_buff_1, %30 ]
  br label %38

85:                                               ; preds = %41
  %86 = phi ptr [ @B_C2_cons_buff_0, %41 ]
  br label %49

87:                                               ; preds = %41
  %88 = phi ptr [ @B_C2_cons_buff_1, %41 ]
  br label %49
}

define void @core_1_2() {
  store i32 0, ptr @_anonymous1
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4)
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8)
  br label %1

1:                                                ; preds = %74, %0
  %2 = phi i64 [ %75, %74 ], [ 0, %0 ]
  %3 = phi i32 [ %9, %74 ], [ 0, %0 ]
  %4 = phi i32 [ %10, %74 ], [ 0, %0 ]
  %5 = phi i32 [ %11, %74 ], [ 0, %0 ]
  %6 = icmp slt i64 %2, 4294967295
  br i1 %6, label %7, label %76

7:                                                ; preds = %66, %1
  %8 = phi i64 [ %73, %66 ], [ 0, %1 ]
  %9 = phi i32 [ %67, %66 ], [ %3, %1 ]
  %10 = phi i32 [ %27, %66 ], [ %4, %1 ]
  %11 = phi i32 [ %28, %66 ], [ %5, %1 ]
  %12 = icmp slt i64 %8, 4
  br i1 %12, label %13, label %74

13:                                               ; preds = %7
  %14 = sub i32 1, %9
  %15 = icmp sgt i32 %14, 0
  %16 = select i1 %15, i32 %14, i32 0
  %17 = sub i32 0, %16
  call void @llvm.aie2p.acquire(i32 52, i32 %17)
  %18 = add i32 %9, %16
  %19 = load i32, ptr @_anonymous1
  %20 = sext i32 %19 to i64
  switch i64 %20, label %21 [
    i64 0, label %77
    i64 1, label %79
  ]

21:                                               ; preds = %77, %79, %13
  %22 = phi ptr [ %80, %79 ], [ %78, %77 ], [ @C_C1_buff_0, %13 ]
  %23 = getelementptr [32 x [128 x i32]], ptr %22, i32 0, i32 0, i32 0
  br label %24

24:                                               ; preds = %21
  call void @zero_i32(ptr %23)
  br label %25

25:                                               ; preds = %52, %24
  %26 = phi i64 [ %65, %52 ], [ 0, %24 ]
  %27 = phi i32 [ %53, %52 ], [ %10, %24 ]
  %28 = phi i32 [ %59, %52 ], [ %11, %24 ]
  %29 = icmp slt i64 %26, 32
  br i1 %29, label %30, label %66

30:                                               ; preds = %25
  %31 = sub i32 1, %27
  %32 = icmp sgt i32 %31, 0
  %33 = select i1 %32, i32 %31, i32 0
  %34 = sub i32 0, %33
  call void @llvm.aie2p.acquire(i32 49, i32 %34)
  %35 = add i32 %27, %33
  %36 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4)
  %37 = sext i32 %36 to i64
  switch i64 %37, label %38 [
    i64 0, label %81
    i64 1, label %83
  ]

38:                                               ; preds = %81, %83, %30
  %39 = phi ptr [ %84, %83 ], [ %82, %81 ], [ @A_C_1_cons_buff_0, %30 ]
  %40 = getelementptr [32 x [64 x i8]], ptr %39, i32 0, i32 0, i32 0
  br label %41

41:                                               ; preds = %38
  %42 = sub i32 1, %28
  %43 = icmp sgt i32 %42, 0
  %44 = select i1 %43, i32 %42, i32 0
  %45 = sub i32 0, %44
  call void @llvm.aie2p.acquire(i32 51, i32 %45)
  %46 = add i32 %28, %44
  %47 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8)
  %48 = sext i32 %47 to i64
  switch i64 %48, label %49 [
    i64 0, label %85
    i64 1, label %87
  ]

49:                                               ; preds = %85, %87, %41
  %50 = phi ptr [ %88, %87 ], [ %86, %85 ], [ @B_C1_cons_buff_0, %41 ]
  %51 = getelementptr [64 x [128 x i8]], ptr %50, i32 0, i32 0, i32 0
  br label %52

52:                                               ; preds = %49
  call void @matmul_i8_i32(ptr %40, ptr %51, ptr %23)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %53 = sub i32 %35, 1
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4)
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %59 = sub i32 %46, 1
  %60 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8)
  %61 = add i32 %60, 1
  %62 = icmp sge i32 %61, 2
  %63 = add i32 %60, -1
  %64 = select i1 %62, i32 %63, i32 %61
  store i32 %64, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8)
  %65 = add i64 %26, 1
  br label %25

66:                                               ; preds = %25
  call void @llvm.aie2p.release(i32 53, i32 1)
  %67 = sub i32 %18, 1
  %68 = load i32, ptr @_anonymous1
  %69 = add i32 %68, 1
  %70 = icmp sge i32 %69, 2
  %71 = add i32 %68, -1
  %72 = select i1 %70, i32 %71, i32 %69
  store i32 %72, ptr @_anonymous1
  %73 = add i64 %8, 1
  br label %7

74:                                               ; preds = %7
  %75 = add i64 %2, 1
  br label %1

76:                                               ; preds = %1
  ret void

77:                                               ; preds = %13
  %78 = phi ptr [ @C_C1_buff_0, %13 ]
  br label %21

79:                                               ; preds = %13
  %80 = phi ptr [ @C_C1_buff_1, %13 ]
  br label %21

81:                                               ; preds = %30
  %82 = phi ptr [ @A_C_1_cons_buff_0, %30 ]
  br label %38

83:                                               ; preds = %30
  %84 = phi ptr [ @A_C_1_cons_buff_1, %30 ]
  br label %38

85:                                               ; preds = %41
  %86 = phi ptr [ @B_C1_cons_buff_0, %41 ]
  br label %49

87:                                               ; preds = %41
  %88 = phi ptr [ @B_C1_cons_buff_1, %41 ]
  br label %49
}

define void @core_0_2() {
  store i32 0, ptr @_anonymous0
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4)
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  br label %1

1:                                                ; preds = %74, %0
  %2 = phi i64 [ %75, %74 ], [ 0, %0 ]
  %3 = phi i32 [ %9, %74 ], [ 0, %0 ]
  %4 = phi i32 [ %10, %74 ], [ 0, %0 ]
  %5 = phi i32 [ %11, %74 ], [ 0, %0 ]
  %6 = icmp slt i64 %2, 4294967295
  br i1 %6, label %7, label %76

7:                                                ; preds = %66, %1
  %8 = phi i64 [ %73, %66 ], [ 0, %1 ]
  %9 = phi i32 [ %67, %66 ], [ %3, %1 ]
  %10 = phi i32 [ %27, %66 ], [ %4, %1 ]
  %11 = phi i32 [ %28, %66 ], [ %5, %1 ]
  %12 = icmp slt i64 %8, 4
  br i1 %12, label %13, label %74

13:                                               ; preds = %7
  %14 = sub i32 1, %9
  %15 = icmp sgt i32 %14, 0
  %16 = select i1 %15, i32 %14, i32 0
  %17 = sub i32 0, %16
  call void @llvm.aie2p.acquire(i32 52, i32 %17)
  %18 = add i32 %9, %16
  %19 = load i32, ptr @_anonymous0
  %20 = sext i32 %19 to i64
  switch i64 %20, label %21 [
    i64 0, label %77
    i64 1, label %79
  ]

21:                                               ; preds = %77, %79, %13
  %22 = phi ptr [ %80, %79 ], [ %78, %77 ], [ @C_C0_buff_0, %13 ]
  %23 = getelementptr [32 x [128 x i32]], ptr %22, i32 0, i32 0, i32 0
  br label %24

24:                                               ; preds = %21
  call void @zero_i32(ptr %23)
  br label %25

25:                                               ; preds = %52, %24
  %26 = phi i64 [ %65, %52 ], [ 0, %24 ]
  %27 = phi i32 [ %53, %52 ], [ %10, %24 ]
  %28 = phi i32 [ %59, %52 ], [ %11, %24 ]
  %29 = icmp slt i64 %26, 32
  br i1 %29, label %30, label %66

30:                                               ; preds = %25
  %31 = sub i32 1, %27
  %32 = icmp sgt i32 %31, 0
  %33 = select i1 %32, i32 %31, i32 0
  %34 = sub i32 0, %33
  call void @llvm.aie2p.acquire(i32 49, i32 %34)
  %35 = add i32 %27, %33
  %36 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4)
  %37 = sext i32 %36 to i64
  switch i64 %37, label %38 [
    i64 0, label %81
    i64 1, label %83
  ]

38:                                               ; preds = %81, %83, %30
  %39 = phi ptr [ %84, %83 ], [ %82, %81 ], [ @A_C_0_cons_buff_0, %30 ]
  %40 = getelementptr [32 x [64 x i8]], ptr %39, i32 0, i32 0, i32 0
  br label %41

41:                                               ; preds = %38
  %42 = sub i32 1, %28
  %43 = icmp sgt i32 %42, 0
  %44 = select i1 %43, i32 %42, i32 0
  %45 = sub i32 0, %44
  call void @llvm.aie2p.acquire(i32 51, i32 %45)
  %46 = add i32 %28, %44
  %47 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  %48 = sext i32 %47 to i64
  switch i64 %48, label %49 [
    i64 0, label %85
    i64 1, label %87
  ]

49:                                               ; preds = %85, %87, %41
  %50 = phi ptr [ %88, %87 ], [ %86, %85 ], [ @B_C0_cons_buff_0, %41 ]
  %51 = getelementptr [64 x [128 x i8]], ptr %50, i32 0, i32 0, i32 0
  br label %52

52:                                               ; preds = %49
  call void @matmul_i8_i32(ptr %40, ptr %51, ptr %23)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %53 = sub i32 %35, 1
  %54 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4)
  %55 = add i32 %54, 1
  %56 = icmp sge i32 %55, 2
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %59 = sub i32 %46, 1
  %60 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  %61 = add i32 %60, 1
  %62 = icmp sge i32 %61, 2
  %63 = add i32 %60, -1
  %64 = select i1 %62, i32 %63, i32 %61
  store i32 %64, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  %65 = add i64 %26, 1
  br label %25

66:                                               ; preds = %25
  call void @llvm.aie2p.release(i32 53, i32 1)
  %67 = sub i32 %18, 1
  %68 = load i32, ptr @_anonymous0
  %69 = add i32 %68, 1
  %70 = icmp sge i32 %69, 2
  %71 = add i32 %68, -1
  %72 = select i1 %70, i32 %71, i32 %69
  store i32 %72, ptr @_anonymous0
  %73 = add i64 %8, 1
  br label %7

74:                                               ; preds = %7
  %75 = add i64 %2, 1
  br label %1

76:                                               ; preds = %1
  ret void

77:                                               ; preds = %13
  %78 = phi ptr [ @C_C0_buff_0, %13 ]
  br label %21

79:                                               ; preds = %13
  %80 = phi ptr [ @C_C0_buff_1, %13 ]
  br label %21

81:                                               ; preds = %30
  %82 = phi ptr [ @A_C_0_cons_buff_0, %30 ]
  br label %38

83:                                               ; preds = %30
  %84 = phi ptr [ @A_C_0_cons_buff_1, %30 ]
  br label %38

85:                                               ; preds = %41
  %86 = phi ptr [ @B_C0_cons_buff_0, %41 ]
  br label %49

87:                                               ; preds = %41
  %88 = phi ptr [ @B_C0_cons_buff_1, %41 ]
  br label %49
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
