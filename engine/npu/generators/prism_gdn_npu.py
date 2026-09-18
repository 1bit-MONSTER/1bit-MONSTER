#!/usr/bin/env python3
# prism_gdn_gen2.py — P4.2: packed-input IRON/AIE design for the GDN conv1d slice.
# One shim (1 MM2S + 1 S2MM, inside the channel limit), one mem tile, one core.
import argparse, numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cd", type=int, default=256)
    a = p.parse_args()
    CD = a.cd
    IN = CD + CD * 4 + CD * 3
    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def body():
            IN_ty  = np.ndarray[(IN,), np.dtype[np.float32]]
            OUT_ty = np.ndarray[(CD,), np.dtype[np.float32]]
            k = external_func("prism_gdn_conv1d_packed", inputs=[IN_ty, OUT_ty],
                              link_with="prism_gdn_packed.o")
            shim = tile(0,0); mem = tile(0,1); core_t = tile(0,2)
            in_s = object_fifo("IN_S", shim, mem, 1, IN_ty)
            in_c = object_fifo("IN_C", mem, core_t, 1, IN_ty)
            o_c  = object_fifo("O_C", core_t, mem, 1, OUT_ty)
            o_s  = object_fifo("O_S", mem, shim, 1, OUT_ty)
            object_fifo_link(in_s, in_c); object_fifo_link(o_c, o_s)
            @core(core_t, stack_size=0x2000)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    x = in_c.acquire(ObjectFifoPort.Consume, 1)
                    o = o_c.acquire(ObjectFifoPort.Produce, 1)
                    k(x, o)
                    in_c.release(ObjectFifoPort.Consume, 1)
                    o_c.release(ObjectFifoPort.Produce, 1)
            @runtime_sequence(np.ndarray[(IN,), np.dtype[np.float32]],
                              np.ndarray[(CD,), np.dtype[np.float32]])
            def seq(X, O):
                t = shim_dma_single_bd_task(in_s, X, offset=0, sizes=[1,1,1,IN], strides=[1,1,1,1], issue_token=True)
                dma_start_task(t); dma_await_task(t); dma_free_task(t)
                t = shim_dma_single_bd_task(o_s, O, offset=0, sizes=[1,1,1,CD], strides=[1,1,1,1], issue_token=True)
                dma_start_task(t); dma_await_task(t); dma_free_task(t)
        print(ctx.module)
main()
