; ModuleID = '/home/bcloud/1bit-systems/engine/npu/generators/clean_g.mlir.prj/peano-compat_main.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous3 = external local_unnamed_addr global [3 x i32]
@_anonymous2 = external local_unnamed_addr global [3 x i32]
@_anonymous1 = external local_unnamed_addr global [3 x i32]
@_anonymous0 = external local_unnamed_addr global [3 x i32]
@A_C_3_cons_buff_1 = external global [32 x [64 x i8]]
@A_C_3_cons_buff_0 = external global [32 x [64 x i8]]
@A_C_2_cons_buff_1 = external global [32 x [64 x i8]]
@A_C_2_cons_buff_0 = external global [32 x [64 x i8]]
@A_C_1_cons_buff_1 = external global [32 x [64 x i8]]
@A_C_1_cons_buff_0 = external global [32 x [64 x i8]]
@A_C_0_cons_buff_1 = external global [32 x [64 x i8]]
@A_C_0_cons_buff_0 = external global [32 x [64 x i8]]
@B_C0_cons_buff_1 = external global [64 x [128 x i8]]
@B_C0_cons_buff_0 = external global [64 x [128 x i8]]
@C_C0_buff_1 = external global [32 x [128 x i32]]
@C_C0_buff_0 = external global [32 x [128 x i32]]
@B_C1_cons_buff_1 = external global [64 x [128 x i8]]
@B_C1_cons_buff_0 = external global [64 x [128 x i8]]
@C_C1_buff_1 = external global [32 x [128 x i32]]
@C_C1_buff_0 = external global [32 x [128 x i32]]
@B_C2_cons_buff_1 = external global [64 x [128 x i8]]
@B_C2_cons_buff_0 = external global [64 x [128 x i8]]
@C_C2_buff_1 = external global [32 x [128 x i32]]
@C_C2_buff_0 = external global [32 x [128 x i32]]
@B_C3_cons_buff_1 = external global [64 x [128 x i8]]
@B_C3_cons_buff_0 = external global [64 x [128 x i8]]
@C_C3_buff_1 = external global [32 x [128 x i32]]
@C_C3_buff_0 = external global [32 x [128 x i32]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @zero_i32(ptr) local_unnamed_addr

declare void @matmul_i8_i32(ptr, ptr, ptr) local_unnamed_addr

define void @core_3_2() local_unnamed_addr {
  store i32 0, ptr @_anonymous3, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  br label %.preheader

.preheader:                                       ; preds = %0, %70
  %1 = phi i64 [ 0, %0 ], [ %71, %70 ]
  br label %2

2:                                                ; preds = %.preheader, %62
  %3 = phi i64 [ 0, %.preheader ], [ %68, %62 ]
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  %4 = load i32, ptr @_anonymous3, align 4
  %cond = icmp eq i32 %4, 1
  %spec.select = select i1 %cond, ptr @C_C3_buff_1, ptr @C_C3_buff_0
  tail call void @zero_i32(ptr nonnull %spec.select)
  br label %5

5:                                                ; preds = %5, %2
  %6 = phi i64 [ 0, %2 ], [ %60, %5 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  %cond1 = icmp eq i32 %7, 1
  %spec.select4 = select i1 %cond1, ptr @A_C_3_cons_buff_1, ptr @A_C_3_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %8 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  %cond2 = icmp eq i32 %8, 1
  %9 = select i1 %cond2, ptr @B_C3_cons_buff_1, ptr @B_C3_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4, ptr nonnull %9, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %10 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  %11 = add i32 %10, 1
  %12 = icmp sgt i32 %11, 1
  %13 = add i32 %10, -1
  %14 = select i1 %12, i32 %13, i32 %11
  store i32 %14, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %15 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  %16 = add i32 %15, 1
  %17 = icmp sgt i32 %16, 1
  %18 = add i32 %15, -1
  %19 = select i1 %17, i32 %18, i32 %16
  store i32 %19, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %20 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  %cond1.1 = icmp eq i32 %20, 1
  %spec.select4.1 = select i1 %cond1.1, ptr @A_C_3_cons_buff_1, ptr @A_C_3_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  %cond2.1 = icmp eq i32 %21, 1
  %22 = select i1 %cond2.1, ptr @B_C3_cons_buff_1, ptr @B_C3_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.1, ptr nonnull %22, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %23 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  %24 = add i32 %23, 1
  %25 = icmp sgt i32 %24, 1
  %26 = add i32 %23, -1
  %27 = select i1 %25, i32 %26, i32 %24
  store i32 %27, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %28 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  %29 = add i32 %28, 1
  %30 = icmp sgt i32 %29, 1
  %31 = add i32 %28, -1
  %32 = select i1 %30, i32 %31, i32 %29
  store i32 %32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  %cond1.2 = icmp eq i32 %33, 1
  %spec.select4.2 = select i1 %cond1.2, ptr @A_C_3_cons_buff_1, ptr @A_C_3_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %34 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  %cond2.2 = icmp eq i32 %34, 1
  %35 = select i1 %cond2.2, ptr @B_C3_cons_buff_1, ptr @B_C3_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.2, ptr nonnull %35, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %36 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  %37 = add i32 %36, 1
  %38 = icmp sgt i32 %37, 1
  %39 = add i32 %36, -1
  %40 = select i1 %38, i32 %39, i32 %37
  store i32 %40, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %41 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  %42 = add i32 %41, 1
  %43 = icmp sgt i32 %42, 1
  %44 = add i32 %41, -1
  %45 = select i1 %43, i32 %44, i32 %42
  store i32 %45, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  %46 = or disjoint i64 %6, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %47 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  %cond1.3 = icmp eq i32 %47, 1
  %spec.select4.3 = select i1 %cond1.3, ptr @A_C_3_cons_buff_1, ptr @A_C_3_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %48 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  %cond2.3 = icmp eq i32 %48, 1
  %49 = select i1 %cond2.3, ptr @B_C3_cons_buff_1, ptr @B_C3_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.3, ptr nonnull %49, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %50 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  %51 = add i32 %50, 1
  %52 = icmp sgt i32 %51, 1
  %53 = add i32 %50, -1
  %54 = select i1 %52, i32 %53, i32 %51
  store i32 %54, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %55 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  %56 = add i32 %55, 1
  %57 = icmp sgt i32 %56, 1
  %58 = add i32 %55, -1
  %59 = select i1 %57, i32 %58, i32 %56
  store i32 %59, ptr getelementptr inbounds nuw (i8, ptr @_anonymous3, i20 8), align 4
  %60 = add nuw nsw i64 %6, 4
  %61 = icmp samesign ult i64 %46, 31
  br i1 %61, label %5, label %62

62:                                               ; preds = %5
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %63 = load i32, ptr @_anonymous3, align 4
  %64 = add i32 %63, 1
  %65 = icmp sgt i32 %64, 1
  %66 = add i32 %63, -1
  %67 = select i1 %65, i32 %66, i32 %64
  store i32 %67, ptr @_anonymous3, align 4
  %68 = add nuw nsw i64 %3, 1
  %69 = icmp samesign ult i64 %3, 3
  br i1 %69, label %2, label %70

70:                                               ; preds = %62
  %71 = add nuw nsw i64 %1, 1
  %72 = icmp samesign ult i64 %1, 4294967294
  br i1 %72, label %.preheader, label %73

73:                                               ; preds = %70
  ret void
}

define void @core_2_2() local_unnamed_addr {
  store i32 0, ptr @_anonymous2, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  br label %.preheader

.preheader:                                       ; preds = %0, %70
  %1 = phi i64 [ 0, %0 ], [ %71, %70 ]
  br label %2

2:                                                ; preds = %.preheader, %62
  %3 = phi i64 [ 0, %.preheader ], [ %68, %62 ]
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  %4 = load i32, ptr @_anonymous2, align 4
  %cond = icmp eq i32 %4, 1
  %spec.select = select i1 %cond, ptr @C_C2_buff_1, ptr @C_C2_buff_0
  tail call void @zero_i32(ptr nonnull %spec.select)
  br label %5

5:                                                ; preds = %5, %2
  %6 = phi i64 [ 0, %2 ], [ %60, %5 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  %cond1 = icmp eq i32 %7, 1
  %spec.select4 = select i1 %cond1, ptr @A_C_2_cons_buff_1, ptr @A_C_2_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %8 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  %cond2 = icmp eq i32 %8, 1
  %9 = select i1 %cond2, ptr @B_C2_cons_buff_1, ptr @B_C2_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4, ptr nonnull %9, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %10 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  %11 = add i32 %10, 1
  %12 = icmp sgt i32 %11, 1
  %13 = add i32 %10, -1
  %14 = select i1 %12, i32 %13, i32 %11
  store i32 %14, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %15 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  %16 = add i32 %15, 1
  %17 = icmp sgt i32 %16, 1
  %18 = add i32 %15, -1
  %19 = select i1 %17, i32 %18, i32 %16
  store i32 %19, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %20 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  %cond1.1 = icmp eq i32 %20, 1
  %spec.select4.1 = select i1 %cond1.1, ptr @A_C_2_cons_buff_1, ptr @A_C_2_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  %cond2.1 = icmp eq i32 %21, 1
  %22 = select i1 %cond2.1, ptr @B_C2_cons_buff_1, ptr @B_C2_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.1, ptr nonnull %22, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %23 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  %24 = add i32 %23, 1
  %25 = icmp sgt i32 %24, 1
  %26 = add i32 %23, -1
  %27 = select i1 %25, i32 %26, i32 %24
  store i32 %27, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %28 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  %29 = add i32 %28, 1
  %30 = icmp sgt i32 %29, 1
  %31 = add i32 %28, -1
  %32 = select i1 %30, i32 %31, i32 %29
  store i32 %32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  %cond1.2 = icmp eq i32 %33, 1
  %spec.select4.2 = select i1 %cond1.2, ptr @A_C_2_cons_buff_1, ptr @A_C_2_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %34 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  %cond2.2 = icmp eq i32 %34, 1
  %35 = select i1 %cond2.2, ptr @B_C2_cons_buff_1, ptr @B_C2_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.2, ptr nonnull %35, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %36 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  %37 = add i32 %36, 1
  %38 = icmp sgt i32 %37, 1
  %39 = add i32 %36, -1
  %40 = select i1 %38, i32 %39, i32 %37
  store i32 %40, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %41 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  %42 = add i32 %41, 1
  %43 = icmp sgt i32 %42, 1
  %44 = add i32 %41, -1
  %45 = select i1 %43, i32 %44, i32 %42
  store i32 %45, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  %46 = or disjoint i64 %6, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %47 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  %cond1.3 = icmp eq i32 %47, 1
  %spec.select4.3 = select i1 %cond1.3, ptr @A_C_2_cons_buff_1, ptr @A_C_2_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %48 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  %cond2.3 = icmp eq i32 %48, 1
  %49 = select i1 %cond2.3, ptr @B_C2_cons_buff_1, ptr @B_C2_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.3, ptr nonnull %49, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %50 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  %51 = add i32 %50, 1
  %52 = icmp sgt i32 %51, 1
  %53 = add i32 %50, -1
  %54 = select i1 %52, i32 %53, i32 %51
  store i32 %54, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %55 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  %56 = add i32 %55, 1
  %57 = icmp sgt i32 %56, 1
  %58 = add i32 %55, -1
  %59 = select i1 %57, i32 %58, i32 %56
  store i32 %59, ptr getelementptr inbounds nuw (i8, ptr @_anonymous2, i20 8), align 4
  %60 = add nuw nsw i64 %6, 4
  %61 = icmp samesign ult i64 %46, 31
  br i1 %61, label %5, label %62

62:                                               ; preds = %5
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %63 = load i32, ptr @_anonymous2, align 4
  %64 = add i32 %63, 1
  %65 = icmp sgt i32 %64, 1
  %66 = add i32 %63, -1
  %67 = select i1 %65, i32 %66, i32 %64
  store i32 %67, ptr @_anonymous2, align 4
  %68 = add nuw nsw i64 %3, 1
  %69 = icmp samesign ult i64 %3, 3
  br i1 %69, label %2, label %70

70:                                               ; preds = %62
  %71 = add nuw nsw i64 %1, 1
  %72 = icmp samesign ult i64 %1, 4294967294
  br i1 %72, label %.preheader, label %73

73:                                               ; preds = %70
  ret void
}

define void @core_1_2() local_unnamed_addr {
  store i32 0, ptr @_anonymous1, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  br label %.preheader

.preheader:                                       ; preds = %0, %70
  %1 = phi i64 [ 0, %0 ], [ %71, %70 ]
  br label %2

2:                                                ; preds = %.preheader, %62
  %3 = phi i64 [ 0, %.preheader ], [ %68, %62 ]
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  %4 = load i32, ptr @_anonymous1, align 4
  %cond = icmp eq i32 %4, 1
  %spec.select = select i1 %cond, ptr @C_C1_buff_1, ptr @C_C1_buff_0
  tail call void @zero_i32(ptr nonnull %spec.select)
  br label %5

5:                                                ; preds = %5, %2
  %6 = phi i64 [ 0, %2 ], [ %60, %5 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  %cond1 = icmp eq i32 %7, 1
  %spec.select4 = select i1 %cond1, ptr @A_C_1_cons_buff_1, ptr @A_C_1_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %8 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  %cond2 = icmp eq i32 %8, 1
  %9 = select i1 %cond2, ptr @B_C1_cons_buff_1, ptr @B_C1_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4, ptr nonnull %9, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %10 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  %11 = add i32 %10, 1
  %12 = icmp sgt i32 %11, 1
  %13 = add i32 %10, -1
  %14 = select i1 %12, i32 %13, i32 %11
  store i32 %14, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %15 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  %16 = add i32 %15, 1
  %17 = icmp sgt i32 %16, 1
  %18 = add i32 %15, -1
  %19 = select i1 %17, i32 %18, i32 %16
  store i32 %19, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %20 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  %cond1.1 = icmp eq i32 %20, 1
  %spec.select4.1 = select i1 %cond1.1, ptr @A_C_1_cons_buff_1, ptr @A_C_1_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  %cond2.1 = icmp eq i32 %21, 1
  %22 = select i1 %cond2.1, ptr @B_C1_cons_buff_1, ptr @B_C1_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.1, ptr nonnull %22, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %23 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  %24 = add i32 %23, 1
  %25 = icmp sgt i32 %24, 1
  %26 = add i32 %23, -1
  %27 = select i1 %25, i32 %26, i32 %24
  store i32 %27, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %28 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  %29 = add i32 %28, 1
  %30 = icmp sgt i32 %29, 1
  %31 = add i32 %28, -1
  %32 = select i1 %30, i32 %31, i32 %29
  store i32 %32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  %cond1.2 = icmp eq i32 %33, 1
  %spec.select4.2 = select i1 %cond1.2, ptr @A_C_1_cons_buff_1, ptr @A_C_1_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %34 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  %cond2.2 = icmp eq i32 %34, 1
  %35 = select i1 %cond2.2, ptr @B_C1_cons_buff_1, ptr @B_C1_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.2, ptr nonnull %35, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %36 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  %37 = add i32 %36, 1
  %38 = icmp sgt i32 %37, 1
  %39 = add i32 %36, -1
  %40 = select i1 %38, i32 %39, i32 %37
  store i32 %40, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %41 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  %42 = add i32 %41, 1
  %43 = icmp sgt i32 %42, 1
  %44 = add i32 %41, -1
  %45 = select i1 %43, i32 %44, i32 %42
  store i32 %45, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  %46 = or disjoint i64 %6, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %47 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  %cond1.3 = icmp eq i32 %47, 1
  %spec.select4.3 = select i1 %cond1.3, ptr @A_C_1_cons_buff_1, ptr @A_C_1_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %48 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  %cond2.3 = icmp eq i32 %48, 1
  %49 = select i1 %cond2.3, ptr @B_C1_cons_buff_1, ptr @B_C1_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.3, ptr nonnull %49, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %50 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  %51 = add i32 %50, 1
  %52 = icmp sgt i32 %51, 1
  %53 = add i32 %50, -1
  %54 = select i1 %52, i32 %53, i32 %51
  store i32 %54, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %55 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  %56 = add i32 %55, 1
  %57 = icmp sgt i32 %56, 1
  %58 = add i32 %55, -1
  %59 = select i1 %57, i32 %58, i32 %56
  store i32 %59, ptr getelementptr inbounds nuw (i8, ptr @_anonymous1, i20 8), align 4
  %60 = add nuw nsw i64 %6, 4
  %61 = icmp samesign ult i64 %46, 31
  br i1 %61, label %5, label %62

62:                                               ; preds = %5
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %63 = load i32, ptr @_anonymous1, align 4
  %64 = add i32 %63, 1
  %65 = icmp sgt i32 %64, 1
  %66 = add i32 %63, -1
  %67 = select i1 %65, i32 %66, i32 %64
  store i32 %67, ptr @_anonymous1, align 4
  %68 = add nuw nsw i64 %3, 1
  %69 = icmp samesign ult i64 %3, 3
  br i1 %69, label %2, label %70

70:                                               ; preds = %62
  %71 = add nuw nsw i64 %1, 1
  %72 = icmp samesign ult i64 %1, 4294967294
  br i1 %72, label %.preheader, label %73

73:                                               ; preds = %70
  ret void
}

define void @core_0_2() local_unnamed_addr {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  br label %.preheader

.preheader:                                       ; preds = %0, %70
  %1 = phi i64 [ 0, %0 ], [ %71, %70 ]
  br label %2

2:                                                ; preds = %.preheader, %62
  %3 = phi i64 [ 0, %.preheader ], [ %68, %62 ]
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  %4 = load i32, ptr @_anonymous0, align 4
  %cond = icmp eq i32 %4, 1
  %spec.select = select i1 %cond, ptr @C_C0_buff_1, ptr @C_C0_buff_0
  tail call void @zero_i32(ptr nonnull %spec.select)
  br label %5

5:                                                ; preds = %5, %2
  %6 = phi i64 [ 0, %2 ], [ %60, %5 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %cond1 = icmp eq i32 %7, 1
  %spec.select4 = select i1 %cond1, ptr @A_C_0_cons_buff_1, ptr @A_C_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %8 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2 = icmp eq i32 %8, 1
  %9 = select i1 %cond2, ptr @B_C0_cons_buff_1, ptr @B_C0_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4, ptr nonnull %9, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %10 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %11 = add i32 %10, 1
  %12 = icmp sgt i32 %11, 1
  %13 = add i32 %10, -1
  %14 = select i1 %12, i32 %13, i32 %11
  store i32 %14, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %15 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %16 = add i32 %15, 1
  %17 = icmp sgt i32 %16, 1
  %18 = add i32 %15, -1
  %19 = select i1 %17, i32 %18, i32 %16
  store i32 %19, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %20 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %cond1.1 = icmp eq i32 %20, 1
  %spec.select4.1 = select i1 %cond1.1, ptr @A_C_0_cons_buff_1, ptr @A_C_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2.1 = icmp eq i32 %21, 1
  %22 = select i1 %cond2.1, ptr @B_C0_cons_buff_1, ptr @B_C0_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.1, ptr nonnull %22, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %23 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %24 = add i32 %23, 1
  %25 = icmp sgt i32 %24, 1
  %26 = add i32 %23, -1
  %27 = select i1 %25, i32 %26, i32 %24
  store i32 %27, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %28 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %29 = add i32 %28, 1
  %30 = icmp sgt i32 %29, 1
  %31 = add i32 %28, -1
  %32 = select i1 %30, i32 %31, i32 %29
  store i32 %32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %cond1.2 = icmp eq i32 %33, 1
  %spec.select4.2 = select i1 %cond1.2, ptr @A_C_0_cons_buff_1, ptr @A_C_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %34 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2.2 = icmp eq i32 %34, 1
  %35 = select i1 %cond2.2, ptr @B_C0_cons_buff_1, ptr @B_C0_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.2, ptr nonnull %35, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %36 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %37 = add i32 %36, 1
  %38 = icmp sgt i32 %37, 1
  %39 = add i32 %36, -1
  %40 = select i1 %38, i32 %39, i32 %37
  store i32 %40, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %41 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %42 = add i32 %41, 1
  %43 = icmp sgt i32 %42, 1
  %44 = add i32 %41, -1
  %45 = select i1 %43, i32 %44, i32 %42
  store i32 %45, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %46 = or disjoint i64 %6, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %47 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %cond1.3 = icmp eq i32 %47, 1
  %spec.select4.3 = select i1 %cond1.3, ptr @A_C_0_cons_buff_1, ptr @A_C_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %48 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2.3 = icmp eq i32 %48, 1
  %49 = select i1 %cond2.3, ptr @B_C0_cons_buff_1, ptr @B_C0_cons_buff_0
  tail call void @matmul_i8_i32(ptr nonnull %spec.select4.3, ptr nonnull %49, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %50 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %51 = add i32 %50, 1
  %52 = icmp sgt i32 %51, 1
  %53 = add i32 %50, -1
  %54 = select i1 %52, i32 %53, i32 %51
  store i32 %54, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %55 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %56 = add i32 %55, 1
  %57 = icmp sgt i32 %56, 1
  %58 = add i32 %55, -1
  %59 = select i1 %57, i32 %58, i32 %56
  store i32 %59, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %60 = add nuw nsw i64 %6, 4
  %61 = icmp samesign ult i64 %46, 31
  br i1 %61, label %5, label %62

62:                                               ; preds = %5
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %63 = load i32, ptr @_anonymous0, align 4
  %64 = add i32 %63, 1
  %65 = icmp sgt i32 %64, 1
  %66 = add i32 %63, -1
  %67 = select i1 %65, i32 %66, i32 %64
  store i32 %67, ptr @_anonymous0, align 4
  %68 = add nuw nsw i64 %3, 1
  %69 = icmp samesign ult i64 %3, 3
  br i1 %69, label %2, label %70

70:                                               ; preds = %62
  %71 = add nuw nsw i64 %1, 1
  %72 = icmp samesign ult i64 %1, 4294967294
  br i1 %72, label %.preheader, label %73

73:                                               ; preds = %70
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
