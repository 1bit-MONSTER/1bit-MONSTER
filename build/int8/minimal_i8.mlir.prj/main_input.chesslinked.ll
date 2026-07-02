; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-p:32:32:32:32:8-s0:512:512-a0:8:8-S512-n32:64-P1-p0:20:32:32:32:8-p1:20:32:32:32:8-p2:20:32:32:32:8-p3:20:32:32:32:8-p4:20:32:32:32:8-p5:20:32:32:32:8-p6:20:32:32:32:8-p7:20:32:32:32:8-p8:20:32:32:32:8-p9:20:32:32:32:8-p10:20:32:32:32:8-p11:20:32:32:32:8-p12:20:32:32:32:8-p13:20:32:32:32:8-p14:20:32:32:32:8-p15:20:32:32:32:8"
target triple = "aie2p"

%struct.ipd.custom_type.uint2_t = type { i2 }
%struct.ipd.custom_type.tm_byte_t = type { i8 }

@_anonymous0 = external global [3 x i32]
@a_c_cons_buff_1 = external global [32 x [64 x i8]]
@a_c_cons_buff_0 = external global [32 x [64 x i8]]
@b_c_cons_buff_1 = external global [64 x [128 x i8]]
@b_c_cons_buff_0 = external global [64 x [128 x i8]]
@c_c_buff_0 = external global [32 x [128 x i16]]

define void @core_0_2() addrspace(0) {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.acquire(i32 52, i32 -1)
  call addrspace(0) void @zero_i16(ptr @c_c_buff_0)
  call addrspace(0) void @llvm.aie2p.acquire(i32 49, i32 -1)
  %1 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %1, label %2 [
    i32 0, label %2
    i32 1, label %3
  ]

2:                                                ; preds = %0, %0
  br label %4

3:                                                ; preds = %0
  br label %4

4:                                                ; preds = %3, %2
  %5 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @a_c_cons_buff_1, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %3 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @a_c_cons_buff_0, i64 0, [2 x i64] [i64 32, i64 64], [2 x i64] [i64 64, i64 1] }, %2 ]
  call addrspace(0) void @llvm.aie2p.acquire(i32 51, i32 -1)
  %6 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %6, label %7 [
    i32 0, label %7
    i32 1, label %8
  ]

7:                                                ; preds = %4, %4
  br label %9

8:                                                ; preds = %4
  br label %9

9:                                                ; preds = %8, %7
  %10 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @b_c_cons_buff_1, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %8 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @b_c_cons_buff_0, i64 0, [2 x i64] [i64 64, i64 128], [2 x i64] [i64 128, i64 1] }, %7 ]
  %11 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 1
  %12 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, 1
  call addrspace(0) void @matmul_i8_i16(ptr %11, ptr %12, ptr @c_c_buff_0)
  call addrspace(0) void @llvm.aie2p.release(i32 48, i32 1)
  %13 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  %14 = add i32 %13, 1
  %15 = icmp sge i32 %14, 2
  %16 = add i32 %13, -1
  %17 = select i1 %15, i32 %16, i32 %14
  store i32 %17, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 50, i32 1)
  %18 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  %19 = add i32 %18, 1
  %20 = icmp sge i32 %19, 2
  %21 = add i32 %18, -1
  %22 = select i1 %20, i32 %21, i32 %19
  store i32 %22, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8), align 4
  call addrspace(0) void @llvm.aie2p.release(i32 53, i32 1)
  %23 = load i32, ptr @_anonymous0, align 4
  %24 = add i32 %23, 1
  %25 = icmp sge i32 %24, 1
  %26 = select i1 %25, i32 %23, i32 %24
  store i32 %26, ptr @_anonymous0, align 4
  ret void
}

declare void @llvm.aie2p.acquire(i32, i32) addrspace(0)

declare void @zero_i16(ptr) addrspace(0)

declare void @matmul_i8_i16(ptr, ptr, ptr) addrspace(0)

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
