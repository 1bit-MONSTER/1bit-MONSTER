// npu_engine_i8ctx_inc.h — I8Ctx GEMM context, extracted verbatim from
// npu_engine_universal.cpp so standalone probes/tools can reuse the exact
// NPU INT8 GEMM path (packB/quantize/launch/dequant). Keep in sync with the
// engine (search "struct I8Ctx{" there).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_elf.h>
struct I8Ctx{int MD,KD,ND,NL;std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;
    std::unique_ptr<xrt::module>mdl;std::unique_ptr<xrt::elf>elf;
    std::unique_ptr<xrt::ext::kernel>k;std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bA,bC;
    std::vector<std::unique_ptr<xrt::bo>>layerB;int8_t*Am;int32_t*Cm;
    std::vector<std::vector<float>> group_scales;
    bool initialized=false;
    ~I8Ctx(){/* Am/Cm are mapped from bA/bC — destroyed by unique_ptr dtors */}
    bool isReady(){return initialized&&k&&bA&&bC;}
    bool init(xrt::device&d,const char*xp,const char*ip,int gid_B,int nlayers){
        NL=nlayers;FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
        ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
        // Convert instructions to ELF module for extended kernel API
        try{
            std::vector<char> iraw((char*)ins.data(),(char*)ins.data()+ins.size()*sizeof(uint32_t));
            aiebu::aiebu_assembler asmblr(aiebu::aiebu_assembler::buffer_type::blob_instr_transaction,iraw);
            auto e=asmblr.get_elf();
            xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);
            hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());
            elf=std::make_unique<xrt::elf>(e.data(),e.size());
        }catch(std::exception&ex){
            fprintf(stderr,"  aiebu ELF gen failed: %s\n",ex.what());return false;}
        mdl=std::make_unique<xrt::module>(*elf);
        k=std::make_unique<xrt::ext::kernel>(*hc,*mdl,"MLIR_AIE");
        // Create data BOs (instruction BO not needed — embedded in ELF)
        bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,0);
        bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*4,XRT_BO_FLAGS_HOST_ONLY,0);
        Am=(int8_t*)bA->map();Cm=(int32_t*)bC->map();
        for(int l=0;l<NL;l++)layerB.emplace_back(std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,0));
        group_scales.resize(NL);initialized=true;return true;}
    // Per-group INT8 quantization: K is divided into groups of 32, each with its own scale.
    // go()/dequantize() compute the effective scale as the average of per-group scales.
    void packB(int l,const float*w,int K,int N,float&sout){int num_groups=(K+31)/32;
        group_scales[l].resize(num_groups);auto*Bm=(int8_t*)layerB[l]->map();
        for(int g=0;g<num_groups;g++){int g_start=g*32;int g_size=std::min(32,K-g_start);float g_amax=0;
            for(int j=0;j<N;j++){for(int i=0;i<g_size;i++){float a=fabsf(w[(g_start+i)*N+j]);if(std::isfinite(a)&&a>g_amax)g_amax=a;}}
            if(g_amax<1e-12f)g_amax=1.0f;group_scales[l][g]=g_amax/127.0f;float g_is=127.0f/g_amax;
            for(int j=0;j<N;j++){for(int i=0;i<g_size;i++){float v=w[(g_start+i)*N+j];if(!std::isfinite(v))v=0;
                int x=(int)roundf(v*g_is);if(x>127)x=127;else if(x<-127)x=-127;Bm[(g_start+i)*N+j]=(int8_t)x;}}}
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);float ssum=0;for(int g=0;g<num_groups;g++)ssum+=group_scales[l][g];sout=ssum/num_groups;}
    // Async quantize: packs float activations into the A buffer without syncing
    // Returns the quantized buffer pointer for later sync_and_launch.
    inline int8_t* quantize_async(const float*A,int am,int ak,float ascale){
        float ais=1.0f/ascale;
        memset(Am,0,(size_t)am*KD);
        for(int m=0;m<am;m++)for(int k=0;k<ak;k++){
            float v=A[m*ak+k];if(!std::isfinite(v))v=0;
            int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;
            Am[m*KD+k]=(int8_t)q;}
        return Am;
    }
    // Sync A to device (non-blocking DMA, can overlap with NPU compute).
    inline void sync_A(int l){
        (void)l;
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    // Launch kernel via extended module API (instructions embedded in ELF).
    // Args: mode=3, ctrl=0, reserved=0, then data BOs: A, weights B, output C.
    inline xrt::run launch(int l){
        return k->operator()(3,0,0,*bA,*layerB[l],*bC);
    }
    // Sync A to device and launch kernel. Returns run handle.
    inline xrt::run sync_and_launch(int l){
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return k->operator()(3,0,0,*bA,*layerB[l],*bC);
    }

    // Wait for run, sync C back, and dequantize.
    // layer: if >= 0, uses average of per-group scales instead of Bscale.
    inline void dequantize(xrt::run& r,float*C,int am,int an,float ascale,float Bscale,int layer=-1){
        r.wait();
        readback();
        dequant_only(C,am,an,ascale,Bscale,layer);
    }
    // Wait for NPU kernel completion without readback.
    // Returns immediately after kernel finishes. Call sync_back_and_dequant() later.
    inline void wait_kernel(xrt::run& r){
        r.wait();
    }
    // Phase-split (cross-layer pipeline, roadmap step 3): bC DMA readback only.
    // Call after wait_kernel(), then dequant_only() or a fused consumer pass.
    inline void readback(){
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }
    // CPU-only dequant loop (call after readback()). CAN overlap with the next
    // kernel's NPU execution. layer: if >= 0, uses average of per-group scales
    // instead of Bscale.
    inline void dequant_only(float*C,int am,int an,float ascale,float Bscale,int layer=-1){
        if(layer>=0&&(size_t)layer<group_scales.size()&&!group_scales[layer].empty()){
            float ssum=0;for(float s:group_scales[layer])ssum+=s;
            Bscale=ssum/group_scales[layer].size();}
        float cs=ascale*Bscale;
        for(int m=0;m<am;m++)for(int n=0;n<an;n++){
            float val=(float)((int32_t)Cm[m*ND+n])*cs;if(!std::isfinite(val))val=0;
            C[m*an+n]=val;}
    }
    // Sync C back from device and dequantize (call after wait_kernel).
    // This is CPU-only work that CAN overlap with the next kernel's NPU execution.
    // layer: if >= 0, uses average of per-group scales instead of Bscale.
    inline void sync_back_and_dequant(float*C,int am,int an,float ascale,float Bscale,int layer=-1){
        readback();
        dequant_only(C,am,an,ascale,Bscale,layer);
    }
    // Synchronous go() — simple, always works
    // Uses per-group scales from group_scales[l] when available.
    inline bool go(int l,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){
        quantize_async(A,am,ak,ascale);
        auto r=sync_and_launch(l);
        r.wait();
        dequantize(r,C,am,an,ascale,Bscale,l);
        return true;
    }
    // Fast path: launch, return run handle for later wait+dequant
    inline xrt::run launch_async(int l,const float*A,int am,int ak,float ascale){
        quantize_async(A,am,ak,ascale);
        return sync_and_launch(l);
    }
    // Complete an async launch: wait + dequant
    // layer: if >= 0, uses average of per-group scales instead of Bscale.
    inline void finish_async(xrt::run& r,float*C,int am,int an,float ascale,float Bscale,int layer=-1){
        r.wait();
        dequantize(r,C,am,an,ascale,Bscale,layer);
    }

    // Alternative init that generates instructions via gemm_generate_sequence_i8().
    // Uses the instruction generator instead of pre-compiled instruction files.
    // Creates all BOs for the GEMM context at the specified dimensions.
    bool init_with_generator(xrt::device& d, const char* xp,
                             int _MD, int _KD, int _ND,
                             int nlayers) {
        MD = _MD; KD = _KD; ND = _ND; NL = nlayers;
        // Generate INT8 instruction sequence
        npu_sequence seq(device_npu2);
        gemm_generate_sequence(&seq, MD, KD, ND, 0, false, 0, 0, 0);
        seq.cmds2seq();
        auto [dp, sz] = seq.dump();
        ins.assign(dp, dp + sz / sizeof(uint32_t));
        // Convert to ELF and create kernel
        try {
            std::vector<char> iraw((char*)ins.data(),
                                   (char*)ins.data() + ins.size() * sizeof(uint32_t));
            aiebu::aiebu_assembler asmblr(
                aiebu::aiebu_assembler::buffer_type::blob_instr_transaction, iraw);
            auto e = asmblr.get_elf();
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            elf = std::make_unique<xrt::elf>(e.data(), e.size());
        } catch (std::exception& ex) {
            fprintf(stderr, "  init_with_generator: aiebu ELF gen failed: %s\n", ex.what());
            return false;
        }
        mdl = std::make_unique<xrt::module>(*elf);
        k = std::make_unique<xrt::ext::kernel>(*hc, *mdl, "MLIR_AIE");
        bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD, XRT_BO_FLAGS_HOST_ONLY, 0);
        bC = std::make_unique<xrt::bo>(d, (size_t)MD * ND * 4, XRT_BO_FLAGS_HOST_ONLY, 0);
        Am = (int8_t*)bA->map();
        Cm = (int32_t*)bC->map();
        for (int l = 0; l < NL; l++)
            layerB.emplace_back(
                std::make_unique<xrt::bo>(d, (size_t)KD * ND, XRT_BO_FLAGS_HOST_ONLY, 0));
        group_scales.resize(NL);
        initialized = true;
        return true;
    }

    // Alternative init that takes pre-generated instructions instead of a file.
    // Attention instructions loaded from pre-compiled file at init.
    bool init_with_instrs(xrt::device& d, const char* xp,
                          const std::vector<uint32_t>& pregen_instrs,
                          int nlayers, int mdim, int kdim, int ndim) {
        NL = nlayers;
        MD = mdim; KD = kdim; ND = ndim;
        ins = pregen_instrs;
        try {
            std::vector<char> iraw((char*)ins.data(),
                                   (char*)ins.data() + ins.size() * sizeof(uint32_t));
            aiebu::aiebu_assembler asmblr(
                aiebu::aiebu_assembler::buffer_type::blob_instr_transaction, iraw);
            auto e = asmblr.get_elf();
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            elf = std::make_unique<xrt::elf>(e.data(), e.size());
        } catch (std::exception& ex) {
            fprintf(stderr, "  aiebu ELF gen failed (instrs): %s\n", ex.what());
            return false;
        }
        mdl = std::make_unique<xrt::module>(*elf);
        k = std::make_unique<xrt::ext::kernel>(*hc, *mdl, "MLIR_AIE");
        // Data BOs created by caller for attention (different layout)
        group_scales.resize(nlayers);
        initialized = true;
        return true;
    }
};

