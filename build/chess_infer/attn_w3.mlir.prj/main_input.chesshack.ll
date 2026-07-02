; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@att_output = external global [256 x i32]
@att_state = external global [8 x i32]
@att_accum = external global [512 x i32]
@att_carrier = external global [40 x i32]
@att_v1 = external global [2048 x i32]
@att_v0 = external global [2048 x i32]
@att_k1 = external global [2048 x i32]
@att_k0 = external global [2048 x i32]
@att_q = external global [256 x i32]

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

declare void @qwen3_attention_bf16_make_carrier_masked(ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32)

declare void @qwen3_attention_bf16_init_accum(ptr, ptr, i32, i32)

declare void @qwen3_attention_bf16_accum_block(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

declare void @qwen3_attention_bf16_finish_accum(ptr, ptr, ptr, i32, i32, i32)

define void @core_0_2() {
  call void @llvm.aie2p.acquire(i32 49, i32 -5)
  call void @qwen3_attention_bf16_init_accum(ptr @att_accum, ptr @att_state, i32 512, i32 8)
  call void @qwen3_attention_bf16_make_carrier_masked(ptr @att_q, ptr @att_k0, ptr @att_carrier, i32 3, i32 3, i32 2, i32 7, i32 256, i32 2048, i32 40)
  call void @qwen3_attention_bf16_accum_block(ptr @att_v0, ptr @att_carrier, ptr @att_accum, ptr @att_state, i32 3, i32 2048, i32 40, i32 512, i32 8)
  call void @qwen3_attention_bf16_make_carrier_masked(ptr @att_q, ptr @att_k1, ptr @att_carrier, i32 3, i32 1, i32 2, i32 7, i32 256, i32 2048, i32 40)
  call void @qwen3_attention_bf16_accum_block(ptr @att_v1, ptr @att_carrier, ptr @att_accum, ptr @att_state, i32 1, i32 2048, i32 40, i32 512, i32 8)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @qwen3_attention_bf16_finish_accum(ptr @att_accum, ptr @att_state, ptr @att_output, i32 256, i32 512, i32 8)
  call void @llvm.aie2p.release(i32 51, i32 1)
  call void @llvm.aie2p.release(i32 48, i32 5)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
