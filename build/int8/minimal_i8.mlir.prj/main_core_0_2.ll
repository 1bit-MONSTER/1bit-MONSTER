; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

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
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @zero_i16(ptr @c_c_buff_0)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @matmul_i8_i16(ptr @a_c_cons_buff_0, ptr @b_c_cons_buff_0, ptr @c_c_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.release(i32 53, i32 1)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
