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
@AC0_cons_buff_0 = external global [32 x [64 x i8]]
@BC0_cons_buff_0 = external global [64 x [128 x i8]]
@CC0_buff_0 = external global [32 x [128 x i16]]
@AC1_cons_buff_0 = external global [32 x [64 x i8]]
@BC1_cons_buff_0 = external global [64 x [128 x i8]]
@CC1_buff_0 = external global [32 x [128 x i16]]
@AC2_cons_buff_0 = external global [32 x [64 x i8]]
@BC2_cons_buff_0 = external global [64 x [128 x i8]]
@CC2_buff_0 = external global [32 x [128 x i16]]
@AC3_cons_buff_0 = external global [32 x [64 x i8]]
@BC3_cons_buff_0 = external global [64 x [128 x i8]]
@CC3_buff_0 = external global [32 x [128 x i16]]
@AC4_cons_buff_0 = external global [32 x [64 x i8]]
@BC4_cons_buff_0 = external global [64 x [128 x i8]]
@CC4_buff_0 = external global [32 x [128 x i16]]
@AC5_cons_buff_0 = external global [32 x [64 x i8]]
@BC5_cons_buff_0 = external global [64 x [128 x i8]]
@CC5_buff_0 = external global [32 x [128 x i16]]
@AC6_cons_buff_0 = external global [32 x [64 x i8]]
@BC6_cons_buff_0 = external global [64 x [128 x i8]]
@CC6_buff_0 = external global [32 x [128 x i16]]
@AC7_cons_buff_0 = external global [32 x [64 x i8]]
@BC7_cons_buff_0 = external global [64 x [128 x i8]]
@CC7_buff_0 = external global [32 x [128 x i16]]

define void @core_7_2() addrspace(0) {
  store i32 0, ptr @_anonymous7, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  br label %1

1:                                                ; preds = %72, %0
  %2 = phi i64 [ %77, %72 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %78

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC7_buff_0)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %17, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %18

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC7_cons_buff_0, ptr @BC7_cons_buff_0, ptr @CC7_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %10 = add i32 %9, 1
  %11 = icmp sge i32 %10, 1
  %12 = select i1 %11, i32 %9, i32 %10
  store i32 %12, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %13 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %14 = add i32 %13, 1
  %15 = icmp sge i32 %14, 1
  %16 = select i1 %15, i32 %13, i32 %14
  store i32 %16, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %17 = add i64 %6, 1
  br label %5

18:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %19 = load i32, ptr @_anonymous7, align 4
  %20 = add i32 %19, 1
  %21 = icmp sge i32 %20, 1
  %22 = select i1 %21, i32 %19, i32 %20
  store i32 %22, ptr @_anonymous7, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC7_buff_0)
  br label %23

23:                                               ; preds = %26, %18
  %24 = phi i64 [ %35, %26 ], [ 0, %18 ]
  %25 = icmp slt i64 %24, 16
  br i1 %25, label %26, label %36

26:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC7_cons_buff_0, ptr @BC7_cons_buff_0, ptr @CC7_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 1
  %30 = select i1 %29, i32 %27, i32 %28
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %31 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %32 = add i32 %31, 1
  %33 = icmp sge i32 %32, 1
  %34 = select i1 %33, i32 %31, i32 %32
  store i32 %34, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %35 = add i64 %24, 1
  br label %23

36:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %37 = load i32, ptr @_anonymous7, align 4
  %38 = add i32 %37, 1
  %39 = icmp sge i32 %38, 1
  %40 = select i1 %39, i32 %37, i32 %38
  store i32 %40, ptr @_anonymous7, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC7_buff_0)
  br label %41

41:                                               ; preds = %44, %36
  %42 = phi i64 [ %53, %44 ], [ 0, %36 ]
  %43 = icmp slt i64 %42, 16
  br i1 %43, label %44, label %54

44:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC7_cons_buff_0, ptr @BC7_cons_buff_0, ptr @CC7_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %45 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %46 = add i32 %45, 1
  %47 = icmp sge i32 %46, 1
  %48 = select i1 %47, i32 %45, i32 %46
  store i32 %48, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %50 = add i32 %49, 1
  %51 = icmp sge i32 %50, 1
  %52 = select i1 %51, i32 %49, i32 %50
  store i32 %52, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %53 = add i64 %42, 1
  br label %41

54:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %55 = load i32, ptr @_anonymous7, align 4
  %56 = add i32 %55, 1
  %57 = icmp sge i32 %56, 1
  %58 = select i1 %57, i32 %55, i32 %56
  store i32 %58, ptr @_anonymous7, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC7_buff_0)
  br label %59

59:                                               ; preds = %62, %54
  %60 = phi i64 [ %71, %62 ], [ 0, %54 ]
  %61 = icmp slt i64 %60, 16
  br i1 %61, label %62, label %72

62:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC7_cons_buff_0, ptr @BC7_cons_buff_0, ptr @CC7_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %63 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  %64 = add i32 %63, 1
  %65 = icmp sge i32 %64, 1
  %66 = select i1 %65, i32 %63, i32 %64
  store i32 %66, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %67 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %68 = add i32 %67, 1
  %69 = icmp sge i32 %68, 1
  %70 = select i1 %69, i32 %67, i32 %68
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8), align 4
  %71 = add i64 %60, 1
  br label %59

72:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %73 = load i32, ptr @_anonymous7, align 4
  %74 = add i32 %73, 1
  %75 = icmp sge i32 %74, 1
  %76 = select i1 %75, i32 %73, i32 %74
  store i32 %76, ptr @_anonymous7, align 4
  %77 = add i64 %2, 1
  br label %1

78:                                               ; preds = %1
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

1:                                                ; preds = %72, %0
  %2 = phi i64 [ %77, %72 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %78

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC6_buff_0)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %17, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %18

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC6_cons_buff_0, ptr @BC6_cons_buff_0, ptr @CC6_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %10 = add i32 %9, 1
  %11 = icmp sge i32 %10, 1
  %12 = select i1 %11, i32 %9, i32 %10
  store i32 %12, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %13 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %14 = add i32 %13, 1
  %15 = icmp sge i32 %14, 1
  %16 = select i1 %15, i32 %13, i32 %14
  store i32 %16, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %17 = add i64 %6, 1
  br label %5

18:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %19 = load i32, ptr @_anonymous6, align 4
  %20 = add i32 %19, 1
  %21 = icmp sge i32 %20, 1
  %22 = select i1 %21, i32 %19, i32 %20
  store i32 %22, ptr @_anonymous6, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC6_buff_0)
  br label %23

23:                                               ; preds = %26, %18
  %24 = phi i64 [ %35, %26 ], [ 0, %18 ]
  %25 = icmp slt i64 %24, 16
  br i1 %25, label %26, label %36

26:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC6_cons_buff_0, ptr @BC6_cons_buff_0, ptr @CC6_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 1
  %30 = select i1 %29, i32 %27, i32 %28
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %31 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %32 = add i32 %31, 1
  %33 = icmp sge i32 %32, 1
  %34 = select i1 %33, i32 %31, i32 %32
  store i32 %34, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %35 = add i64 %24, 1
  br label %23

36:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %37 = load i32, ptr @_anonymous6, align 4
  %38 = add i32 %37, 1
  %39 = icmp sge i32 %38, 1
  %40 = select i1 %39, i32 %37, i32 %38
  store i32 %40, ptr @_anonymous6, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC6_buff_0)
  br label %41

41:                                               ; preds = %44, %36
  %42 = phi i64 [ %53, %44 ], [ 0, %36 ]
  %43 = icmp slt i64 %42, 16
  br i1 %43, label %44, label %54

44:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC6_cons_buff_0, ptr @BC6_cons_buff_0, ptr @CC6_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %45 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %46 = add i32 %45, 1
  %47 = icmp sge i32 %46, 1
  %48 = select i1 %47, i32 %45, i32 %46
  store i32 %48, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %50 = add i32 %49, 1
  %51 = icmp sge i32 %50, 1
  %52 = select i1 %51, i32 %49, i32 %50
  store i32 %52, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %53 = add i64 %42, 1
  br label %41

54:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %55 = load i32, ptr @_anonymous6, align 4
  %56 = add i32 %55, 1
  %57 = icmp sge i32 %56, 1
  %58 = select i1 %57, i32 %55, i32 %56
  store i32 %58, ptr @_anonymous6, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC6_buff_0)
  br label %59

59:                                               ; preds = %62, %54
  %60 = phi i64 [ %71, %62 ], [ 0, %54 ]
  %61 = icmp slt i64 %60, 16
  br i1 %61, label %62, label %72

62:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC6_cons_buff_0, ptr @BC6_cons_buff_0, ptr @CC6_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %63 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  %64 = add i32 %63, 1
  %65 = icmp sge i32 %64, 1
  %66 = select i1 %65, i32 %63, i32 %64
  store i32 %66, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %67 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %68 = add i32 %67, 1
  %69 = icmp sge i32 %68, 1
  %70 = select i1 %69, i32 %67, i32 %68
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous6, i64 8), align 4
  %71 = add i64 %60, 1
  br label %59

72:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %73 = load i32, ptr @_anonymous6, align 4
  %74 = add i32 %73, 1
  %75 = icmp sge i32 %74, 1
  %76 = select i1 %75, i32 %73, i32 %74
  store i32 %76, ptr @_anonymous6, align 4
  %77 = add i64 %2, 1
  br label %1

78:                                               ; preds = %1
  ret void
}

define void @core_5_2() addrspace(0) {
  store i32 0, ptr @_anonymous5, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  br label %1

1:                                                ; preds = %72, %0
  %2 = phi i64 [ %77, %72 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %78

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC5_buff_0)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %17, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %18

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC5_cons_buff_0, ptr @BC5_cons_buff_0, ptr @CC5_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %10 = add i32 %9, 1
  %11 = icmp sge i32 %10, 1
  %12 = select i1 %11, i32 %9, i32 %10
  store i32 %12, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %13 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %14 = add i32 %13, 1
  %15 = icmp sge i32 %14, 1
  %16 = select i1 %15, i32 %13, i32 %14
  store i32 %16, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %17 = add i64 %6, 1
  br label %5

18:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %19 = load i32, ptr @_anonymous5, align 4
  %20 = add i32 %19, 1
  %21 = icmp sge i32 %20, 1
  %22 = select i1 %21, i32 %19, i32 %20
  store i32 %22, ptr @_anonymous5, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC5_buff_0)
  br label %23

23:                                               ; preds = %26, %18
  %24 = phi i64 [ %35, %26 ], [ 0, %18 ]
  %25 = icmp slt i64 %24, 16
  br i1 %25, label %26, label %36

26:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC5_cons_buff_0, ptr @BC5_cons_buff_0, ptr @CC5_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 1
  %30 = select i1 %29, i32 %27, i32 %28
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %31 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %32 = add i32 %31, 1
  %33 = icmp sge i32 %32, 1
  %34 = select i1 %33, i32 %31, i32 %32
  store i32 %34, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %35 = add i64 %24, 1
  br label %23

36:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %37 = load i32, ptr @_anonymous5, align 4
  %38 = add i32 %37, 1
  %39 = icmp sge i32 %38, 1
  %40 = select i1 %39, i32 %37, i32 %38
  store i32 %40, ptr @_anonymous5, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC5_buff_0)
  br label %41

41:                                               ; preds = %44, %36
  %42 = phi i64 [ %53, %44 ], [ 0, %36 ]
  %43 = icmp slt i64 %42, 16
  br i1 %43, label %44, label %54

44:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC5_cons_buff_0, ptr @BC5_cons_buff_0, ptr @CC5_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %45 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %46 = add i32 %45, 1
  %47 = icmp sge i32 %46, 1
  %48 = select i1 %47, i32 %45, i32 %46
  store i32 %48, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %50 = add i32 %49, 1
  %51 = icmp sge i32 %50, 1
  %52 = select i1 %51, i32 %49, i32 %50
  store i32 %52, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %53 = add i64 %42, 1
  br label %41

54:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %55 = load i32, ptr @_anonymous5, align 4
  %56 = add i32 %55, 1
  %57 = icmp sge i32 %56, 1
  %58 = select i1 %57, i32 %55, i32 %56
  store i32 %58, ptr @_anonymous5, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC5_buff_0)
  br label %59

59:                                               ; preds = %62, %54
  %60 = phi i64 [ %71, %62 ], [ 0, %54 ]
  %61 = icmp slt i64 %60, 16
  br i1 %61, label %62, label %72

62:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC5_cons_buff_0, ptr @BC5_cons_buff_0, ptr @CC5_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %63 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  %64 = add i32 %63, 1
  %65 = icmp sge i32 %64, 1
  %66 = select i1 %65, i32 %63, i32 %64
  store i32 %66, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %67 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %68 = add i32 %67, 1
  %69 = icmp sge i32 %68, 1
  %70 = select i1 %69, i32 %67, i32 %68
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous5, i64 8), align 4
  %71 = add i64 %60, 1
  br label %59

72:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %73 = load i32, ptr @_anonymous5, align 4
  %74 = add i32 %73, 1
  %75 = icmp sge i32 %74, 1
  %76 = select i1 %75, i32 %73, i32 %74
  store i32 %76, ptr @_anonymous5, align 4
  %77 = add i64 %2, 1
  br label %1

78:                                               ; preds = %1
  ret void
}

define void @core_4_2() addrspace(0) {
  store i32 0, ptr @_anonymous4, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  br label %1

1:                                                ; preds = %72, %0
  %2 = phi i64 [ %77, %72 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %78

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC4_buff_0)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %17, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %18

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC4_cons_buff_0, ptr @BC4_cons_buff_0, ptr @CC4_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %10 = add i32 %9, 1
  %11 = icmp sge i32 %10, 1
  %12 = select i1 %11, i32 %9, i32 %10
  store i32 %12, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %13 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %14 = add i32 %13, 1
  %15 = icmp sge i32 %14, 1
  %16 = select i1 %15, i32 %13, i32 %14
  store i32 %16, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %17 = add i64 %6, 1
  br label %5

18:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %19 = load i32, ptr @_anonymous4, align 4
  %20 = add i32 %19, 1
  %21 = icmp sge i32 %20, 1
  %22 = select i1 %21, i32 %19, i32 %20
  store i32 %22, ptr @_anonymous4, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC4_buff_0)
  br label %23

23:                                               ; preds = %26, %18
  %24 = phi i64 [ %35, %26 ], [ 0, %18 ]
  %25 = icmp slt i64 %24, 16
  br i1 %25, label %26, label %36

26:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC4_cons_buff_0, ptr @BC4_cons_buff_0, ptr @CC4_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 1
  %30 = select i1 %29, i32 %27, i32 %28
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %31 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %32 = add i32 %31, 1
  %33 = icmp sge i32 %32, 1
  %34 = select i1 %33, i32 %31, i32 %32
  store i32 %34, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %35 = add i64 %24, 1
  br label %23

36:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %37 = load i32, ptr @_anonymous4, align 4
  %38 = add i32 %37, 1
  %39 = icmp sge i32 %38, 1
  %40 = select i1 %39, i32 %37, i32 %38
  store i32 %40, ptr @_anonymous4, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC4_buff_0)
  br label %41

41:                                               ; preds = %44, %36
  %42 = phi i64 [ %53, %44 ], [ 0, %36 ]
  %43 = icmp slt i64 %42, 16
  br i1 %43, label %44, label %54

44:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC4_cons_buff_0, ptr @BC4_cons_buff_0, ptr @CC4_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %45 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %46 = add i32 %45, 1
  %47 = icmp sge i32 %46, 1
  %48 = select i1 %47, i32 %45, i32 %46
  store i32 %48, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %50 = add i32 %49, 1
  %51 = icmp sge i32 %50, 1
  %52 = select i1 %51, i32 %49, i32 %50
  store i32 %52, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %53 = add i64 %42, 1
  br label %41

54:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %55 = load i32, ptr @_anonymous4, align 4
  %56 = add i32 %55, 1
  %57 = icmp sge i32 %56, 1
  %58 = select i1 %57, i32 %55, i32 %56
  store i32 %58, ptr @_anonymous4, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC4_buff_0)
  br label %59

59:                                               ; preds = %62, %54
  %60 = phi i64 [ %71, %62 ], [ 0, %54 ]
  %61 = icmp slt i64 %60, 16
  br i1 %61, label %62, label %72

62:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC4_cons_buff_0, ptr @BC4_cons_buff_0, ptr @CC4_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %63 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  %64 = add i32 %63, 1
  %65 = icmp sge i32 %64, 1
  %66 = select i1 %65, i32 %63, i32 %64
  store i32 %66, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %67 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %68 = add i32 %67, 1
  %69 = icmp sge i32 %68, 1
  %70 = select i1 %69, i32 %67, i32 %68
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous4, i64 8), align 4
  %71 = add i64 %60, 1
  br label %59

72:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %73 = load i32, ptr @_anonymous4, align 4
  %74 = add i32 %73, 1
  %75 = icmp sge i32 %74, 1
  %76 = select i1 %75, i32 %73, i32 %74
  store i32 %76, ptr @_anonymous4, align 4
  %77 = add i64 %2, 1
  br label %1

78:                                               ; preds = %1
  ret void
}

define void @core_3_2() addrspace(0) {
  store i32 0, ptr @_anonymous3, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  br label %1

1:                                                ; preds = %72, %0
  %2 = phi i64 [ %77, %72 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %78

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC3_buff_0)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %17, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %18

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC3_cons_buff_0, ptr @BC3_cons_buff_0, ptr @CC3_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %10 = add i32 %9, 1
  %11 = icmp sge i32 %10, 1
  %12 = select i1 %11, i32 %9, i32 %10
  store i32 %12, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %13 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %14 = add i32 %13, 1
  %15 = icmp sge i32 %14, 1
  %16 = select i1 %15, i32 %13, i32 %14
  store i32 %16, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %17 = add i64 %6, 1
  br label %5

18:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %19 = load i32, ptr @_anonymous3, align 4
  %20 = add i32 %19, 1
  %21 = icmp sge i32 %20, 1
  %22 = select i1 %21, i32 %19, i32 %20
  store i32 %22, ptr @_anonymous3, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC3_buff_0)
  br label %23

23:                                               ; preds = %26, %18
  %24 = phi i64 [ %35, %26 ], [ 0, %18 ]
  %25 = icmp slt i64 %24, 16
  br i1 %25, label %26, label %36

26:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC3_cons_buff_0, ptr @BC3_cons_buff_0, ptr @CC3_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 1
  %30 = select i1 %29, i32 %27, i32 %28
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %31 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %32 = add i32 %31, 1
  %33 = icmp sge i32 %32, 1
  %34 = select i1 %33, i32 %31, i32 %32
  store i32 %34, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %35 = add i64 %24, 1
  br label %23

36:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %37 = load i32, ptr @_anonymous3, align 4
  %38 = add i32 %37, 1
  %39 = icmp sge i32 %38, 1
  %40 = select i1 %39, i32 %37, i32 %38
  store i32 %40, ptr @_anonymous3, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC3_buff_0)
  br label %41

41:                                               ; preds = %44, %36
  %42 = phi i64 [ %53, %44 ], [ 0, %36 ]
  %43 = icmp slt i64 %42, 16
  br i1 %43, label %44, label %54

44:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC3_cons_buff_0, ptr @BC3_cons_buff_0, ptr @CC3_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %45 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %46 = add i32 %45, 1
  %47 = icmp sge i32 %46, 1
  %48 = select i1 %47, i32 %45, i32 %46
  store i32 %48, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %50 = add i32 %49, 1
  %51 = icmp sge i32 %50, 1
  %52 = select i1 %51, i32 %49, i32 %50
  store i32 %52, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %53 = add i64 %42, 1
  br label %41

54:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %55 = load i32, ptr @_anonymous3, align 4
  %56 = add i32 %55, 1
  %57 = icmp sge i32 %56, 1
  %58 = select i1 %57, i32 %55, i32 %56
  store i32 %58, ptr @_anonymous3, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC3_buff_0)
  br label %59

59:                                               ; preds = %62, %54
  %60 = phi i64 [ %71, %62 ], [ 0, %54 ]
  %61 = icmp slt i64 %60, 16
  br i1 %61, label %62, label %72

62:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC3_cons_buff_0, ptr @BC3_cons_buff_0, ptr @CC3_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %63 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  %64 = add i32 %63, 1
  %65 = icmp sge i32 %64, 1
  %66 = select i1 %65, i32 %63, i32 %64
  store i32 %66, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %67 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %68 = add i32 %67, 1
  %69 = icmp sge i32 %68, 1
  %70 = select i1 %69, i32 %67, i32 %68
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous3, i64 8), align 4
  %71 = add i64 %60, 1
  br label %59

72:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %73 = load i32, ptr @_anonymous3, align 4
  %74 = add i32 %73, 1
  %75 = icmp sge i32 %74, 1
  %76 = select i1 %75, i32 %73, i32 %74
  store i32 %76, ptr @_anonymous3, align 4
  %77 = add i64 %2, 1
  br label %1

78:                                               ; preds = %1
  ret void
}

define void @core_2_2() addrspace(0) {
  store i32 0, ptr @_anonymous2, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  br label %1

1:                                                ; preds = %72, %0
  %2 = phi i64 [ %77, %72 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %78

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC2_buff_0)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %17, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %18

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC2_cons_buff_0, ptr @BC2_cons_buff_0, ptr @CC2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %10 = add i32 %9, 1
  %11 = icmp sge i32 %10, 1
  %12 = select i1 %11, i32 %9, i32 %10
  store i32 %12, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %13 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %14 = add i32 %13, 1
  %15 = icmp sge i32 %14, 1
  %16 = select i1 %15, i32 %13, i32 %14
  store i32 %16, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %17 = add i64 %6, 1
  br label %5

18:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %19 = load i32, ptr @_anonymous2, align 4
  %20 = add i32 %19, 1
  %21 = icmp sge i32 %20, 1
  %22 = select i1 %21, i32 %19, i32 %20
  store i32 %22, ptr @_anonymous2, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC2_buff_0)
  br label %23

23:                                               ; preds = %26, %18
  %24 = phi i64 [ %35, %26 ], [ 0, %18 ]
  %25 = icmp slt i64 %24, 16
  br i1 %25, label %26, label %36

26:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC2_cons_buff_0, ptr @BC2_cons_buff_0, ptr @CC2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 1
  %30 = select i1 %29, i32 %27, i32 %28
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %31 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %32 = add i32 %31, 1
  %33 = icmp sge i32 %32, 1
  %34 = select i1 %33, i32 %31, i32 %32
  store i32 %34, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %35 = add i64 %24, 1
  br label %23

36:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %37 = load i32, ptr @_anonymous2, align 4
  %38 = add i32 %37, 1
  %39 = icmp sge i32 %38, 1
  %40 = select i1 %39, i32 %37, i32 %38
  store i32 %40, ptr @_anonymous2, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC2_buff_0)
  br label %41

41:                                               ; preds = %44, %36
  %42 = phi i64 [ %53, %44 ], [ 0, %36 ]
  %43 = icmp slt i64 %42, 16
  br i1 %43, label %44, label %54

44:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC2_cons_buff_0, ptr @BC2_cons_buff_0, ptr @CC2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %45 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %46 = add i32 %45, 1
  %47 = icmp sge i32 %46, 1
  %48 = select i1 %47, i32 %45, i32 %46
  store i32 %48, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %50 = add i32 %49, 1
  %51 = icmp sge i32 %50, 1
  %52 = select i1 %51, i32 %49, i32 %50
  store i32 %52, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %53 = add i64 %42, 1
  br label %41

54:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %55 = load i32, ptr @_anonymous2, align 4
  %56 = add i32 %55, 1
  %57 = icmp sge i32 %56, 1
  %58 = select i1 %57, i32 %55, i32 %56
  store i32 %58, ptr @_anonymous2, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC2_buff_0)
  br label %59

59:                                               ; preds = %62, %54
  %60 = phi i64 [ %71, %62 ], [ 0, %54 ]
  %61 = icmp slt i64 %60, 16
  br i1 %61, label %62, label %72

62:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC2_cons_buff_0, ptr @BC2_cons_buff_0, ptr @CC2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %63 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  %64 = add i32 %63, 1
  %65 = icmp sge i32 %64, 1
  %66 = select i1 %65, i32 %63, i32 %64
  store i32 %66, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %67 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %68 = add i32 %67, 1
  %69 = icmp sge i32 %68, 1
  %70 = select i1 %69, i32 %67, i32 %68
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous2, i64 8), align 4
  %71 = add i64 %60, 1
  br label %59

72:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %73 = load i32, ptr @_anonymous2, align 4
  %74 = add i32 %73, 1
  %75 = icmp sge i32 %74, 1
  %76 = select i1 %75, i32 %73, i32 %74
  store i32 %76, ptr @_anonymous2, align 4
  %77 = add i64 %2, 1
  br label %1

78:                                               ; preds = %1
  ret void
}

define void @core_1_2() addrspace(0) {
  store i32 0, ptr @_anonymous1, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  br label %1

1:                                                ; preds = %72, %0
  %2 = phi i64 [ %77, %72 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %78

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC1_buff_0)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %17, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %18

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC1_cons_buff_0, ptr @BC1_cons_buff_0, ptr @CC1_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %10 = add i32 %9, 1
  %11 = icmp sge i32 %10, 1
  %12 = select i1 %11, i32 %9, i32 %10
  store i32 %12, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %13 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %14 = add i32 %13, 1
  %15 = icmp sge i32 %14, 1
  %16 = select i1 %15, i32 %13, i32 %14
  store i32 %16, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %17 = add i64 %6, 1
  br label %5

18:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %19 = load i32, ptr @_anonymous1, align 4
  %20 = add i32 %19, 1
  %21 = icmp sge i32 %20, 1
  %22 = select i1 %21, i32 %19, i32 %20
  store i32 %22, ptr @_anonymous1, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC1_buff_0)
  br label %23

23:                                               ; preds = %26, %18
  %24 = phi i64 [ %35, %26 ], [ 0, %18 ]
  %25 = icmp slt i64 %24, 16
  br i1 %25, label %26, label %36

26:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC1_cons_buff_0, ptr @BC1_cons_buff_0, ptr @CC1_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 1
  %30 = select i1 %29, i32 %27, i32 %28
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %31 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %32 = add i32 %31, 1
  %33 = icmp sge i32 %32, 1
  %34 = select i1 %33, i32 %31, i32 %32
  store i32 %34, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %35 = add i64 %24, 1
  br label %23

36:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %37 = load i32, ptr @_anonymous1, align 4
  %38 = add i32 %37, 1
  %39 = icmp sge i32 %38, 1
  %40 = select i1 %39, i32 %37, i32 %38
  store i32 %40, ptr @_anonymous1, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC1_buff_0)
  br label %41

41:                                               ; preds = %44, %36
  %42 = phi i64 [ %53, %44 ], [ 0, %36 ]
  %43 = icmp slt i64 %42, 16
  br i1 %43, label %44, label %54

44:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC1_cons_buff_0, ptr @BC1_cons_buff_0, ptr @CC1_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %45 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %46 = add i32 %45, 1
  %47 = icmp sge i32 %46, 1
  %48 = select i1 %47, i32 %45, i32 %46
  store i32 %48, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %50 = add i32 %49, 1
  %51 = icmp sge i32 %50, 1
  %52 = select i1 %51, i32 %49, i32 %50
  store i32 %52, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %53 = add i64 %42, 1
  br label %41

54:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %55 = load i32, ptr @_anonymous1, align 4
  %56 = add i32 %55, 1
  %57 = icmp sge i32 %56, 1
  %58 = select i1 %57, i32 %55, i32 %56
  store i32 %58, ptr @_anonymous1, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC1_buff_0)
  br label %59

59:                                               ; preds = %62, %54
  %60 = phi i64 [ %71, %62 ], [ 0, %54 ]
  %61 = icmp slt i64 %60, 16
  br i1 %61, label %62, label %72

62:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC1_cons_buff_0, ptr @BC1_cons_buff_0, ptr @CC1_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %63 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  %64 = add i32 %63, 1
  %65 = icmp sge i32 %64, 1
  %66 = select i1 %65, i32 %63, i32 %64
  store i32 %66, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %67 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %68 = add i32 %67, 1
  %69 = icmp sge i32 %68, 1
  %70 = select i1 %69, i32 %67, i32 %68
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous1, i64 8), align 4
  %71 = add i64 %60, 1
  br label %59

72:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %73 = load i32, ptr @_anonymous1, align 4
  %74 = add i32 %73, 1
  %75 = icmp sge i32 %74, 1
  %76 = select i1 %75, i32 %73, i32 %74
  store i32 %76, ptr @_anonymous1, align 4
  %77 = add i64 %2, 1
  br label %1

78:                                               ; preds = %1
  ret void
}

define void @core_0_2() addrspace(0) {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  br label %1

1:                                                ; preds = %72, %0
  %2 = phi i64 [ %77, %72 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %78

4:                                                ; preds = %1
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC0_buff_0)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %17, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 16
  br i1 %7, label %8, label %18

8:                                                ; preds = %5
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC0_cons_buff_0, ptr @BC0_cons_buff_0, ptr @CC0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %9 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %10 = add i32 %9, 1
  %11 = icmp sge i32 %10, 1
  %12 = select i1 %11, i32 %9, i32 %10
  store i32 %12, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %13 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %14 = add i32 %13, 1
  %15 = icmp sge i32 %14, 1
  %16 = select i1 %15, i32 %13, i32 %14
  store i32 %16, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %17 = add i64 %6, 1
  br label %5

18:                                               ; preds = %5
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %19 = load i32, ptr @_anonymous0, align 4
  %20 = add i32 %19, 1
  %21 = icmp sge i32 %20, 1
  %22 = select i1 %21, i32 %19, i32 %20
  store i32 %22, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC0_buff_0)
  br label %23

23:                                               ; preds = %26, %18
  %24 = phi i64 [ %35, %26 ], [ 0, %18 ]
  %25 = icmp slt i64 %24, 16
  br i1 %25, label %26, label %36

26:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC0_cons_buff_0, ptr @BC0_cons_buff_0, ptr @CC0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 1
  %30 = select i1 %29, i32 %27, i32 %28
  store i32 %30, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %31 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %32 = add i32 %31, 1
  %33 = icmp sge i32 %32, 1
  %34 = select i1 %33, i32 %31, i32 %32
  store i32 %34, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %35 = add i64 %24, 1
  br label %23

36:                                               ; preds = %23
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %37 = load i32, ptr @_anonymous0, align 4
  %38 = add i32 %37, 1
  %39 = icmp sge i32 %38, 1
  %40 = select i1 %39, i32 %37, i32 %38
  store i32 %40, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC0_buff_0)
  br label %41

41:                                               ; preds = %44, %36
  %42 = phi i64 [ %53, %44 ], [ 0, %36 ]
  %43 = icmp slt i64 %42, 16
  br i1 %43, label %44, label %54

44:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC0_cons_buff_0, ptr @BC0_cons_buff_0, ptr @CC0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %45 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %46 = add i32 %45, 1
  %47 = icmp sge i32 %46, 1
  %48 = select i1 %47, i32 %45, i32 %46
  store i32 %48, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %49 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %50 = add i32 %49, 1
  %51 = icmp sge i32 %50, 1
  %52 = select i1 %51, i32 %49, i32 %50
  store i32 %52, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %53 = add i64 %42, 1
  br label %41

54:                                               ; preds = %41
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %55 = load i32, ptr @_anonymous0, align 4
  %56 = add i32 %55, 1
  %57 = icmp sge i32 %56, 1
  %58 = select i1 %57, i32 %55, i32 %56
  store i32 %58, ptr @_anonymous0, align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_scalar_i16(ptr @CC0_buff_0)
  br label %59

59:                                               ; preds = %62, %54
  %60 = phi i64 [ %71, %62 ], [ 0, %54 ]
  %61 = icmp slt i64 %60, 16
  br i1 %61, label %62, label %72

62:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @matmul_scalar_i8_i16(ptr @AC0_cons_buff_0, ptr @BC0_cons_buff_0, ptr @CC0_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %63 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %64 = add i32 %63, 1
  %65 = icmp sge i32 %64, 1
  %66 = select i1 %65, i32 %63, i32 %64
  store i32 %66, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %67 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %68 = add i32 %67, 1
  %69 = icmp sge i32 %68, 1
  %70 = select i1 %69, i32 %67, i32 %68
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %71 = add i64 %60, 1
  br label %59

72:                                               ; preds = %59
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %73 = load i32, ptr @_anonymous0, align 4
  %74 = add i32 %73, 1
  %75 = icmp sge i32 %74, 1
  %76 = select i1 %75, i32 %73, i32 %74
  store i32 %76, ptr @_anonymous0, align 4
  %77 = add i64 %2, 1
  br label %1

78:                                               ; preds = %1
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
