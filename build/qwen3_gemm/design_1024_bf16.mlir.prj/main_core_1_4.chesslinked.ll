; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-p:32:32:32:32:8-s0:512:512-a0:8:8-S512-n32:64-P1-p0:20:32:32:32:8-p1:20:32:32:32:8-p2:20:32:32:32:8-p3:20:32:32:32:8-p4:20:32:32:32:8-p5:20:32:32:32:8-p6:20:32:32:32:8-p7:20:32:32:32:8-p8:20:32:32:32:8-p9:20:32:32:32:8-p10:20:32:32:32:8-p11:20:32:32:32:8-p12:20:32:32:32:8-p13:20:32:32:32:8-p14:20:32:32:32:8-p15:20:32:32:32:8"
target triple = "aie2p"

%struct.ipd.custom_type.uint2_t = type { i2 }
%struct.ipd.custom_type.tm_byte_t = type { i8 }

@AL1_2_1_cons_buff_1 = external global [32 x [64 x bfloat]]
@AL1_2_1_cons_buff_0 = external global [32 x [64 x bfloat]]
@BL1_1_2_cons_buff_0 = external global [64 x [128 x bfloat]]
@CL1_1_2_buff_0 = external global [128 x [128 x bfloat]]

define void @core_1_4() addrspace(0) {
  br label %1

1:                                                ; preds = %15, %0
  %2 = phi i64 [ %16, %15 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %17

4:                                                ; preds = %13, %1
  %5 = phi i64 [ %14, %13 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 2
  br i1 %6, label %7, label %15

7:                                                ; preds = %4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_kernel_bf16(ptr @CL1_1_2_buff_0)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %12, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 16
  br i1 %10, label %11, label %13

11:                                               ; preds = %8
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @matmul_vectorized_bf16(ptr @AL1_2_1_cons_buff_0, ptr @BL1_1_2_cons_buff_0, ptr @CL1_1_2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @matmul_vectorized_bf16(ptr @AL1_2_1_cons_buff_1, ptr @BL1_1_2_cons_buff_0, ptr @CL1_1_2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @matmul_vectorized_bf16(ptr @AL1_2_1_cons_buff_0, ptr @BL1_1_2_cons_buff_0, ptr @CL1_1_2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @matmul_vectorized_bf16(ptr @AL1_2_1_cons_buff_1, ptr @BL1_1_2_cons_buff_0, ptr @CL1_1_2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @matmul_vectorized_bf16(ptr @AL1_2_1_cons_buff_0, ptr @BL1_1_2_cons_buff_0, ptr @CL1_1_2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @matmul_vectorized_bf16(ptr @AL1_2_1_cons_buff_1, ptr @BL1_1_2_cons_buff_0, ptr @CL1_1_2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @matmul_vectorized_bf16(ptr @AL1_2_1_cons_buff_0, ptr @BL1_1_2_cons_buff_0, ptr @CL1_1_2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  call addrspace(0) void @matmul_vectorized_bf16(ptr @AL1_2_1_cons_buff_1, ptr @BL1_1_2_cons_buff_0, ptr @CL1_1_2_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %12 = add i64 %9, 2
  br label %8

13:                                               ; preds = %8
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %14 = add i64 %5, 1
  br label %4

15:                                               ; preds = %4
  %16 = add i64 %2, 1
  br label %1

17:                                               ; preds = %1
  ret void
}

declare void @llvm.aie2p.acquire(i32, i32) addrspace(0)

declare void @zero_kernel_bf16(ptr) addrspace(0)

declare void @matmul_vectorized_bf16(ptr, ptr, ptr) addrspace(0)

declare void @llvm.aie2p.release(i32, i32) addrspace(0)

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
