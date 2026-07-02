; ModuleID = 'QKV.mlir.prj/main_core_6_2.peano-compat.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous6 = external local_unnamed_addr global [3 x i32]
@A_L2L1_0_6_cons_buff_1 = external global [32 x [64 x i8]]
@A_L2L1_0_6_cons_buff_0 = external global [32 x [64 x i8]]
@B_L2L1_6_cons_buff_1 = external global [64 x [128 x i8]]
@B_L2L1_6_cons_buff_0 = external global [64 x [128 x i8]]
@C_L1L2_6_0_buff_0 = external global [128 x [128 x i16]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @zero_scalar_i16(ptr) local_unnamed_addr

declare void @matmul_scalar_i8_i16(ptr, ptr, ptr) local_unnamed_addr

define void @core_6_2() local_unnamed_addr {
  store i32 0, ptr @_anonymous6, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  br label %1

1:                                                ; preds = %0, %310
  %2 = phi i64 [ 0, %0 ], [ %315, %310 ]
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @zero_scalar_i16(ptr nonnull @C_L1L2_6_0_buff_0)
  br label %3

3:                                                ; preds = %3, %1
  %4 = phi i64 [ 0, %1 ], [ %74, %3 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %5 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond = icmp eq i32 %5, 1
  %spec.select = select i1 %cond, ptr @B_L2L1_6_cons_buff_1, ptr @B_L2L1_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond1 = icmp eq i32 %6, 1
  %7 = select i1 %cond1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %7, ptr nonnull %spec.select, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %8 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %9 = add i32 %8, 1
  %10 = icmp sgt i32 %9, 1
  %11 = add i32 %8, -1
  %12 = select i1 %10, i32 %11, i32 %9
  store i32 %12, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %13 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond2 = icmp eq i32 %13, 1
  %14 = select i1 %cond2, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %14, ptr nonnull %spec.select, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %15 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %16 = add i32 %15, 1
  %17 = icmp sgt i32 %16, 1
  %18 = add i32 %15, -1
  %19 = select i1 %17, i32 %18, i32 %16
  store i32 %19, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %20 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond3 = icmp eq i32 %20, 1
  %21 = select i1 %cond3, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %21, ptr nonnull %spec.select, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %22 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %23 = add i32 %22, 1
  %24 = icmp sgt i32 %23, 1
  %25 = add i32 %22, -1
  %26 = select i1 %24, i32 %25, i32 %23
  store i32 %26, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond4 = icmp eq i32 %27, 1
  %28 = select i1 %cond4, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %28, ptr nonnull %spec.select, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %29 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %30 = add i32 %29, 1
  %31 = icmp sgt i32 %30, 1
  %32 = add i32 %29, -1
  %33 = select i1 %31, i32 %32, i32 %30
  store i32 %33, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %34 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %35 = add i32 %34, 1
  %36 = icmp sgt i32 %35, 1
  %37 = add i32 %34, -1
  %38 = select i1 %36, i32 %37, i32 %35
  store i32 %38, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %39 = or disjoint i64 %4, 1
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %40 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond.1 = icmp eq i32 %40, 1
  %spec.select.1 = select i1 %cond.1, ptr @B_L2L1_6_cons_buff_1, ptr @B_L2L1_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %41 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond1.1 = icmp eq i32 %41, 1
  %42 = select i1 %cond1.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %42, ptr nonnull %spec.select.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %43 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %44 = add i32 %43, 1
  %45 = icmp sgt i32 %44, 1
  %46 = add i32 %43, -1
  %47 = select i1 %45, i32 %46, i32 %44
  store i32 %47, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %48 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond2.1 = icmp eq i32 %48, 1
  %49 = select i1 %cond2.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %49, ptr nonnull %spec.select.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %50 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %51 = add i32 %50, 1
  %52 = icmp sgt i32 %51, 1
  %53 = add i32 %50, -1
  %54 = select i1 %52, i32 %53, i32 %51
  store i32 %54, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %55 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond3.1 = icmp eq i32 %55, 1
  %56 = select i1 %cond3.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %56, ptr nonnull %spec.select.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %57 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %58 = add i32 %57, 1
  %59 = icmp sgt i32 %58, 1
  %60 = add i32 %57, -1
  %61 = select i1 %59, i32 %60, i32 %58
  store i32 %61, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %62 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond4.1 = icmp eq i32 %62, 1
  %63 = select i1 %cond4.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %63, ptr nonnull %spec.select.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %64 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %65 = add i32 %64, 1
  %66 = icmp sgt i32 %65, 1
  %67 = add i32 %64, -1
  %68 = select i1 %66, i32 %67, i32 %65
  store i32 %68, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %69 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %70 = add i32 %69, 1
  %71 = icmp sgt i32 %70, 1
  %72 = add i32 %69, -1
  %73 = select i1 %71, i32 %72, i32 %70
  store i32 %73, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %74 = add nuw nsw i64 %4, 2
  %75 = icmp samesign ult i64 %39, 15
  br i1 %75, label %3, label %76

76:                                               ; preds = %3
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %77 = load i32, ptr @_anonymous6, align 4
  %78 = icmp ugt i32 %77, 2147483646
  %79 = zext i1 %78 to i32
  %80 = add i32 %77, %79
  store i32 %80, ptr @_anonymous6, align 4
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @zero_scalar_i16(ptr nonnull @C_L1L2_6_0_buff_0)
  br label %81

81:                                               ; preds = %81, %76
  %82 = phi i64 [ 0, %76 ], [ %152, %81 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %83 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond5 = icmp eq i32 %83, 1
  %spec.select24 = select i1 %cond5, ptr @B_L2L1_6_cons_buff_1, ptr @B_L2L1_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %84 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond6 = icmp eq i32 %84, 1
  %85 = select i1 %cond6, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %85, ptr nonnull %spec.select24, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %86 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %87 = add i32 %86, 1
  %88 = icmp sgt i32 %87, 1
  %89 = add i32 %86, -1
  %90 = select i1 %88, i32 %89, i32 %87
  store i32 %90, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %91 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond7 = icmp eq i32 %91, 1
  %92 = select i1 %cond7, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %92, ptr nonnull %spec.select24, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %93 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %94 = add i32 %93, 1
  %95 = icmp sgt i32 %94, 1
  %96 = add i32 %93, -1
  %97 = select i1 %95, i32 %96, i32 %94
  store i32 %97, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %98 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond8 = icmp eq i32 %98, 1
  %99 = select i1 %cond8, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %99, ptr nonnull %spec.select24, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %100 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %101 = add i32 %100, 1
  %102 = icmp sgt i32 %101, 1
  %103 = add i32 %100, -1
  %104 = select i1 %102, i32 %103, i32 %101
  store i32 %104, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %105 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond9 = icmp eq i32 %105, 1
  %106 = select i1 %cond9, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %106, ptr nonnull %spec.select24, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %107 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %108 = add i32 %107, 1
  %109 = icmp sgt i32 %108, 1
  %110 = add i32 %107, -1
  %111 = select i1 %109, i32 %110, i32 %108
  store i32 %111, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %112 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %113 = add i32 %112, 1
  %114 = icmp sgt i32 %113, 1
  %115 = add i32 %112, -1
  %116 = select i1 %114, i32 %115, i32 %113
  store i32 %116, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %117 = or disjoint i64 %82, 1
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %118 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond5.1 = icmp eq i32 %118, 1
  %spec.select24.1 = select i1 %cond5.1, ptr @B_L2L1_6_cons_buff_1, ptr @B_L2L1_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %119 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond6.1 = icmp eq i32 %119, 1
  %120 = select i1 %cond6.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %120, ptr nonnull %spec.select24.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %121 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %122 = add i32 %121, 1
  %123 = icmp sgt i32 %122, 1
  %124 = add i32 %121, -1
  %125 = select i1 %123, i32 %124, i32 %122
  store i32 %125, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %126 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond7.1 = icmp eq i32 %126, 1
  %127 = select i1 %cond7.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %127, ptr nonnull %spec.select24.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %128 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %129 = add i32 %128, 1
  %130 = icmp sgt i32 %129, 1
  %131 = add i32 %128, -1
  %132 = select i1 %130, i32 %131, i32 %129
  store i32 %132, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %133 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond8.1 = icmp eq i32 %133, 1
  %134 = select i1 %cond8.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %134, ptr nonnull %spec.select24.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %135 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %136 = add i32 %135, 1
  %137 = icmp sgt i32 %136, 1
  %138 = add i32 %135, -1
  %139 = select i1 %137, i32 %138, i32 %136
  store i32 %139, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %140 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond9.1 = icmp eq i32 %140, 1
  %141 = select i1 %cond9.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %141, ptr nonnull %spec.select24.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %142 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %143 = add i32 %142, 1
  %144 = icmp sgt i32 %143, 1
  %145 = add i32 %142, -1
  %146 = select i1 %144, i32 %145, i32 %143
  store i32 %146, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %147 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %148 = add i32 %147, 1
  %149 = icmp sgt i32 %148, 1
  %150 = add i32 %147, -1
  %151 = select i1 %149, i32 %150, i32 %148
  store i32 %151, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %152 = add nuw nsw i64 %82, 2
  %153 = icmp samesign ult i64 %117, 15
  br i1 %153, label %81, label %154

154:                                              ; preds = %81
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %155 = load i32, ptr @_anonymous6, align 4
  %156 = icmp ugt i32 %155, 2147483646
  %157 = zext i1 %156 to i32
  %158 = add i32 %155, %157
  store i32 %158, ptr @_anonymous6, align 4
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @zero_scalar_i16(ptr nonnull @C_L1L2_6_0_buff_0)
  br label %159

159:                                              ; preds = %159, %154
  %160 = phi i64 [ 0, %154 ], [ %230, %159 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %161 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond10 = icmp eq i32 %161, 1
  %spec.select25 = select i1 %cond10, ptr @B_L2L1_6_cons_buff_1, ptr @B_L2L1_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %162 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond11 = icmp eq i32 %162, 1
  %163 = select i1 %cond11, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %163, ptr nonnull %spec.select25, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %164 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %165 = add i32 %164, 1
  %166 = icmp sgt i32 %165, 1
  %167 = add i32 %164, -1
  %168 = select i1 %166, i32 %167, i32 %165
  store i32 %168, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %169 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond12 = icmp eq i32 %169, 1
  %170 = select i1 %cond12, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %170, ptr nonnull %spec.select25, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %171 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %172 = add i32 %171, 1
  %173 = icmp sgt i32 %172, 1
  %174 = add i32 %171, -1
  %175 = select i1 %173, i32 %174, i32 %172
  store i32 %175, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %176 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond13 = icmp eq i32 %176, 1
  %177 = select i1 %cond13, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %177, ptr nonnull %spec.select25, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %178 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %179 = add i32 %178, 1
  %180 = icmp sgt i32 %179, 1
  %181 = add i32 %178, -1
  %182 = select i1 %180, i32 %181, i32 %179
  store i32 %182, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %183 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond14 = icmp eq i32 %183, 1
  %184 = select i1 %cond14, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %184, ptr nonnull %spec.select25, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %185 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %186 = add i32 %185, 1
  %187 = icmp sgt i32 %186, 1
  %188 = add i32 %185, -1
  %189 = select i1 %187, i32 %188, i32 %186
  store i32 %189, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %190 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %191 = add i32 %190, 1
  %192 = icmp sgt i32 %191, 1
  %193 = add i32 %190, -1
  %194 = select i1 %192, i32 %193, i32 %191
  store i32 %194, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %195 = or disjoint i64 %160, 1
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %196 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond10.1 = icmp eq i32 %196, 1
  %spec.select25.1 = select i1 %cond10.1, ptr @B_L2L1_6_cons_buff_1, ptr @B_L2L1_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %197 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond11.1 = icmp eq i32 %197, 1
  %198 = select i1 %cond11.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %198, ptr nonnull %spec.select25.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %199 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %200 = add i32 %199, 1
  %201 = icmp sgt i32 %200, 1
  %202 = add i32 %199, -1
  %203 = select i1 %201, i32 %202, i32 %200
  store i32 %203, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %204 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond12.1 = icmp eq i32 %204, 1
  %205 = select i1 %cond12.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %205, ptr nonnull %spec.select25.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %206 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %207 = add i32 %206, 1
  %208 = icmp sgt i32 %207, 1
  %209 = add i32 %206, -1
  %210 = select i1 %208, i32 %209, i32 %207
  store i32 %210, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %211 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond13.1 = icmp eq i32 %211, 1
  %212 = select i1 %cond13.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %212, ptr nonnull %spec.select25.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %213 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %214 = add i32 %213, 1
  %215 = icmp sgt i32 %214, 1
  %216 = add i32 %213, -1
  %217 = select i1 %215, i32 %216, i32 %214
  store i32 %217, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %218 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond14.1 = icmp eq i32 %218, 1
  %219 = select i1 %cond14.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %219, ptr nonnull %spec.select25.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %220 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %221 = add i32 %220, 1
  %222 = icmp sgt i32 %221, 1
  %223 = add i32 %220, -1
  %224 = select i1 %222, i32 %223, i32 %221
  store i32 %224, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %225 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %226 = add i32 %225, 1
  %227 = icmp sgt i32 %226, 1
  %228 = add i32 %225, -1
  %229 = select i1 %227, i32 %228, i32 %226
  store i32 %229, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %230 = add nuw nsw i64 %160, 2
  %231 = icmp samesign ult i64 %195, 15
  br i1 %231, label %159, label %232

232:                                              ; preds = %159
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %233 = load i32, ptr @_anonymous6, align 4
  %234 = icmp ugt i32 %233, 2147483646
  %235 = zext i1 %234 to i32
  %236 = add i32 %233, %235
  store i32 %236, ptr @_anonymous6, align 4
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @zero_scalar_i16(ptr nonnull @C_L1L2_6_0_buff_0)
  br label %237

237:                                              ; preds = %237, %232
  %238 = phi i64 [ 0, %232 ], [ %308, %237 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %239 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond15 = icmp eq i32 %239, 1
  %spec.select26 = select i1 %cond15, ptr @B_L2L1_6_cons_buff_1, ptr @B_L2L1_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %240 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond16 = icmp eq i32 %240, 1
  %241 = select i1 %cond16, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %241, ptr nonnull %spec.select26, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %242 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %243 = add i32 %242, 1
  %244 = icmp sgt i32 %243, 1
  %245 = add i32 %242, -1
  %246 = select i1 %244, i32 %245, i32 %243
  store i32 %246, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %247 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond17 = icmp eq i32 %247, 1
  %248 = select i1 %cond17, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %248, ptr nonnull %spec.select26, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %249 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %250 = add i32 %249, 1
  %251 = icmp sgt i32 %250, 1
  %252 = add i32 %249, -1
  %253 = select i1 %251, i32 %252, i32 %250
  store i32 %253, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %254 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond18 = icmp eq i32 %254, 1
  %255 = select i1 %cond18, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %255, ptr nonnull %spec.select26, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %256 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %257 = add i32 %256, 1
  %258 = icmp sgt i32 %257, 1
  %259 = add i32 %256, -1
  %260 = select i1 %258, i32 %259, i32 %257
  store i32 %260, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %261 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond19 = icmp eq i32 %261, 1
  %262 = select i1 %cond19, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %262, ptr nonnull %spec.select26, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %263 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %264 = add i32 %263, 1
  %265 = icmp sgt i32 %264, 1
  %266 = add i32 %263, -1
  %267 = select i1 %265, i32 %266, i32 %264
  store i32 %267, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %268 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %269 = add i32 %268, 1
  %270 = icmp sgt i32 %269, 1
  %271 = add i32 %268, -1
  %272 = select i1 %270, i32 %271, i32 %269
  store i32 %272, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %273 = or disjoint i64 %238, 1
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %274 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond15.1 = icmp eq i32 %274, 1
  %spec.select26.1 = select i1 %cond15.1, ptr @B_L2L1_6_cons_buff_1, ptr @B_L2L1_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %275 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond16.1 = icmp eq i32 %275, 1
  %276 = select i1 %cond16.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %276, ptr nonnull %spec.select26.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %277 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %278 = add i32 %277, 1
  %279 = icmp sgt i32 %278, 1
  %280 = add i32 %277, -1
  %281 = select i1 %279, i32 %280, i32 %278
  store i32 %281, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %282 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond17.1 = icmp eq i32 %282, 1
  %283 = select i1 %cond17.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %283, ptr nonnull %spec.select26.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %284 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %285 = add i32 %284, 1
  %286 = icmp sgt i32 %285, 1
  %287 = add i32 %284, -1
  %288 = select i1 %286, i32 %287, i32 %285
  store i32 %288, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %289 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond18.1 = icmp eq i32 %289, 1
  %290 = select i1 %cond18.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %290, ptr nonnull %spec.select26.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %291 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %292 = add i32 %291, 1
  %293 = icmp sgt i32 %292, 1
  %294 = add i32 %291, -1
  %295 = select i1 %293, i32 %294, i32 %292
  store i32 %295, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %296 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond19.1 = icmp eq i32 %296, 1
  %297 = select i1 %cond19.1, ptr @A_L2L1_0_6_cons_buff_1, ptr @A_L2L1_0_6_cons_buff_0
  tail call void @matmul_scalar_i8_i16(ptr nonnull %297, ptr nonnull %spec.select26.1, ptr nonnull @C_L1L2_6_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %298 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %299 = add i32 %298, 1
  %300 = icmp sgt i32 %299, 1
  %301 = add i32 %298, -1
  %302 = select i1 %300, i32 %301, i32 %299
  store i32 %302, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %303 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %304 = add i32 %303, 1
  %305 = icmp sgt i32 %304, 1
  %306 = add i32 %303, -1
  %307 = select i1 %305, i32 %306, i32 %304
  store i32 %307, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %308 = add nuw nsw i64 %238, 2
  %309 = icmp samesign ult i64 %273, 15
  br i1 %309, label %237, label %310

310:                                              ; preds = %237
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %311 = load i32, ptr @_anonymous6, align 4
  %312 = icmp ugt i32 %311, 2147483646
  %313 = zext i1 %312 to i32
  %314 = add i32 %311, %313
  store i32 %314, ptr @_anonymous6, align 4
  %315 = add nuw nsw i64 %2, 1
  %316 = icmp samesign ult i64 %2, 4294967294
  br i1 %316, label %1, label %317

317:                                              ; preds = %310
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
