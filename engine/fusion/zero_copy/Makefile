# Zero-copy NPU<->GPU<->host substrate + real parallel proof.
#
# Build:  make test_parallel_real     # real NPU+GPU parallel compute proof
#         make                        # libsharedbo.a + substrate proofs
# Run:    sudo ./test_parallel_real   # need accel + render access
#         ./test_zero_copy            # substrate proof
#         ./test_pipeline             # pipeline timing
#         sudo ./test_vk_dma_buf_import  # Vulkan dma-buf import proof (#1215)
#
# Needs: XRT (2.21+), TheRock ROCm HIP, Vulkan SDK, /dev/accel/accel0.

# TheRock SDK layout: devel tree is a python-site-packages subdir.
THEROCK ?= /opt/rocm-therock
DEVEL    = $(THEROCK)/lib/python3.14/site-packages/_rocm_sdk_devel
XRTINC  ?= /usr/include/xrt
XRTLIB  ?= /usr/lib/x86_64-linux-gnu
CXX      = $(THEROCK)/bin/amdclang++
VK_LIBS  = -lvulkan

CXXFLAGS = -std=c++20 -O2 -g -Wall -Wno-unused-parameter -Wno-unused-result -Wno-unused-value \
           -D__HIP_PLATFORM_AMD__ \
           -I. -I$(XRTINC) -I$(DEVEL)/include
LDFLAGS  = -L$(XRTLIB) -L$(DEVEL)/lib \
           -lxrt_coreutil -luuid -ldl -lpthread -lamdhip64

# dma-buf import SPIR-V (compiled from GLSL at build time)
SHADER_DIR = shaders
FILL_SPV   = $(SHADER_DIR)/fill_pattern.spv
RT_SPV     = $(SHADER_DIR)/roundtrip.spv

.PHONY: all clean

all: libsharedbo.a test_zero_copy test_pipeline test_vk_dma_buf_import

libsharedbo.a: shared_bo.o
	ar rcs $@ $^

# ---- HIP-compiled substrate proofs (C++ only, HIP runtime linked) ----
shared_bo.o: shared_bo.cpp shared_bo.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

test_zero_copy: test_zero_copy.cpp shared_bo.o
	$(CXX) $(CXXFLAGS) $< shared_bo.o -o $@ $(LDFLAGS)

test_pipeline: test_pipeline.cpp pipeline_overlap.cpp pipeline_overlap.h shared_bo.o
	$(CXX) $(CXXFLAGS) $< pipeline_overlap.cpp shared_bo.o -o $@ $(LDFLAGS)

# ---- Vulkan dma-buf import proof (#1215/#1217) ----
$(FILL_SPV): shaders/fill_pattern.comp
	glslc shaders/fill_pattern.comp -o $@

$(RT_SPV): shaders/roundtrip.comp
	glslc shaders/roundtrip.comp -o $@

test_vk_dma_buf_import: test_vk_dma_buf_import.cpp shared_bo.o $(FILL_SPV) $(RT_SPV)
	$(CXX) $(CXXFLAGS) $< shared_bo.o -o $@ $(LDFLAGS) $(VK_LIBS)

# ---- vkrt (src/vulkan_rt.h) production import path — as used by src/backend_fused.cpp ----
test_vkrt_dma_buf_import: test_vkrt_dma_buf_import.cpp shared_bo.o $(FILL_SPV)
	$(CXX) $(CXXFLAGS) -I../../../src $< shared_bo.o -o $@ $(LDFLAGS) $(VK_LIBS)

# ---- zero-copy GPU-attention slice: shader reads NPU SharedBO KV via dma-buf import ----
ATTN_SPV   = $(SHADER_DIR)/attn_decode.spv

$(ATTN_SPV): shaders/attn_decode.comp
	glslc shaders/attn_decode.comp -o $@

test_vk_attn_slice: test_vk_attn_slice.cpp shared_bo.o $(ATTN_SPV)
	$(CXX) $(CXXFLAGS) -I../../../src $< shared_bo.o -o $@ $(LDFLAGS) $(VK_LIBS)

# ---- TRUE parallel proof: HIP __global__ kernel + NPU xclbin on same pages ----
test_parallel_real: test_parallel_real.hip shared_bo.o shared_bo.h
	$(CXX) $(CXXFLAGS) $< shared_bo.o -o $@ $(LDFLAGS)

clean:
	rm -f *.o *.a test_zero_copy test_pipeline test_parallel_real test_vk_dma_buf_import test_vkrt_dma_buf_import test_vk_attn_slice $(FILL_SPV) $(RT_SPV) $(ATTN_SPV)
