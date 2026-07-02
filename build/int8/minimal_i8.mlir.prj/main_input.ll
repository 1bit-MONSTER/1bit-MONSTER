; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@_anonymous0 = external global [3 x i32]
@a_s_cons_buff_1 = external global [32 x [64 x i8]]
@a_s_cons_buff_0 = external global [32 x [64 x i8]]
@a_c_cons_buff_1 = external global [32 x [64 x i8]]
@a_c_cons_buff_0 = external global [32 x [64 x i8]]
@b_s_cons_buff_1 = external global [64 x [128 x i8]]
@b_s_cons_buff_0 = external global [64 x [128 x i8]]
@b_c_cons_buff_1 = external global [64 x [128 x i8]]
@b_c_cons_buff_0 = external global [64 x [128 x i8]]
@c_c_buff_0 = external global [32 x [128 x i16]]
@c_c_cons_buff_0 = external global [32 x [128 x i16]]

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

declare void @matmul_i8_i16(ptr, ptr, ptr)

declare void @zero_i16(ptr)

define void @core_0_2() {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_i16(ptr @c_c_buff_0)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %1, label %2 [
    i32 0, label %2
    i32 1, label %3
  ]

2:                                                ; preds = %0, %0
  br label %4

3:                                                ; preds = %0
  br label %4

4:                                                ; preds = %2, %3
  %5 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @a_c_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %3 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @a_c_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %2 ]
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %6 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %6, label %7 [
    i32 0, label %7
    i32 1, label %8
  ]

7:                                                ; preds = %4, %4
  br label %9

8:                                                ; preds = %4
  br label %9

9:                                                ; preds = %7, %8
  %10 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @b_c_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %8 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @b_c_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %7 ]
  %11 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 1
  %12 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, 1
  call void @matmul_i8_i16(ptr %11, ptr %12, ptr @c_c_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %13 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  %14 = add i32 %13, 1
  %15 = icmp sge i32 %14, 2
  %16 = add i32 %13, -1
  %17 = select i1 %15, i32 %16, i32 %14
  store i32 %17, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  call void @llvm.aie2p.release(i32 50, i32 1)
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  %19 = add i32 %18, 1
  %20 = icmp sge i32 %19, 2
  %21 = add i32 %18, -1
  %22 = select i1 %20, i32 %21, i32 %19
  store i32 %22, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  call void @llvm.aie2p.release(i32 53, i32 1)
  %23 = load i32, ptr @_anonymous0, align 4
  %24 = add i32 %23, 1
  %25 = icmp sge i32 %24, 1
  %26 = select i1 %25, i32 %23, i32 %24
  store i32 %26, ptr @_anonymous0, align 4
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
