; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@A_in_cons_buff_0 = external global [128 x [512 x bfloat]]
@A_l1_cons_buff_0 = external global [32 x [64 x bfloat]]
@B_in_cons_buff_0 = external global [64 x [128 x bfloat]]
@B_l1_cons_buff_0 = external global [64 x [128 x bfloat]]
@C_l1_buff_0 = external global [128 x [128 x bfloat]]
@C_l1_cons_buff_0 = external global [128 x [128 x bfloat]]

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

declare void @zero_kernel_bf16(ptr)

declare void @matmul_vectorized_bf16(ptr, ptr, ptr)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %15, %0
  %2 = phi i64 [ %16, %15 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %17

4:                                                ; preds = %13, %1
  %5 = phi i64 [ %14, %13 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 64
  br i1 %6, label %7, label %15

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_kernel_bf16(ptr @C_l1_buff_0)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %12, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 16
  br i1 %10, label %11, label %13

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @matmul_vectorized_bf16(ptr @A_l1_cons_buff_0, ptr @B_l1_cons_buff_0, ptr @C_l1_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @matmul_vectorized_bf16(ptr @A_l1_cons_buff_0, ptr @B_l1_cons_buff_0, ptr @C_l1_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @matmul_vectorized_bf16(ptr @A_l1_cons_buff_0, ptr @B_l1_cons_buff_0, ptr @C_l1_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @matmul_vectorized_bf16(ptr @A_l1_cons_buff_0, ptr @B_l1_cons_buff_0, ptr @C_l1_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %12 = add i64 %9, 1
  br label %8

13:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 53, i32 1)
  %14 = add i64 %5, 1
  br label %4

15:                                               ; preds = %4
  %16 = add i64 %2, 1
  br label %1

17:                                               ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
