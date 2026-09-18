#!/usr/bin/env python3
# SiLU-only: GU (bf16) -> silu_gate_up -> silu (bf16). 1 core.
import argparse, numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=16); p.add_argument("-IM", type=int, default=128)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        M, IM = a.M, a.IM
        @device(AIEDevice.npu2)
        def body():
            GU_ty = np.ndarray[(M, 2*IM), np.dtype[bfloat16]]
            SL_ty = np.ndarray[(M, IM), np.dtype[bfloat16]]
            silu = external_func("silu_gate_up", inputs=[GU_ty, SL_ty], link_with="silu_gate_up.o")
            shim = tile(0,0); mem = tile(0,1); core_t = tile(0,2)
            GU_s = object_fifo("GU_S", shim, mem, 2, GU_ty)
            GU_c = object_fifo("GU_C", mem, core_t, 2, GU_ty)
            SL_c = object_fifo("SL_C", core_t, mem, 2, SL_ty)
            SL_s = object_fifo("SL_S", mem, shim, 2, SL_ty)
            object_fifo_link(GU_s, GU_c)
            object_fifo_link(SL_c, SL_s)
            @core(core_t, stack_size=0x2000)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(1):
                        gu = GU_c.acquire(ObjectFifoPort.Consume, 1)
                        sl = SL_c.acquire(ObjectFifoPort.Produce, 1)
                        silu(gu, sl)
                        GU_c.release(ObjectFifoPort.Consume, 1)
                        SL_c.release(ObjectFifoPort.Produce, 1)
            @runtime_sequence(np.ndarray[(M*2*IM,), np.dtype[bfloat16]], np.ndarray[(M*IM,), np.dtype[bfloat16]])
            def seq(GU, SL):
                gt = shim_dma_single_bd_task(GU_s, GU, offset=0, sizes=[1,1,1,M*2*IM], strides=[1,1,1,1], issue_token=True)
                dma_start_task(gt); dma_await_task(gt); dma_free_task(gt)
                st = shim_dma_single_bd_task(SL_s, SL, offset=0, sizes=[1,1,1,M*IM], strides=[1,1,1,1], issue_token=True)
                dma_start_task(st); dma_await_task(st); dma_free_task(st)
        print(ctx.module)
main()
