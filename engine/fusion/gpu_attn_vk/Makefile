# gpu_attn_vk — Vulkan in-place attention engine for the fused NPU-FFN path.
# Build:  make            (shaders + test_vk_attn_layer)
# Run:    ./test_vk_attn_layer   (needs /dev/accel/accel0 + renderD128)
THEROCK ?= /opt/rocm-therock
DEVEL    = $(THEROCK)/lib/python3.14/site-packages/_rocm_sdk_devel
XRTINC  ?= /usr/include/xrt
XRTLIB  ?= /usr/lib/x86_64-linux-gnu
CXX      = $(THEROCK)/bin/amdclang++
VK_LIBS  = -lvulkan

CXXFLAGS = -std=c++20 -O2 -g -D__HIP_PLATFORM_AMD__ -Wall -Wno-unused-parameter -Wno-unused-result \
           -I. -I../../src -I../zero_copy -I$(XRTINC) -I$(DEVEL)/include
LDFLAGS  = -L$(XRTLIB) -L$(DEVEL)/lib \
           -lxrt_coreutil -luuid -ldl -lpthread -lamdhip64

SPVS = $(patsubst shaders/%.comp, shaders/%.spv, $(wildcard shaders/*.comp))

.PHONY: all clean

all: $(SPVS) test_vk_attn_layer

shaders/%.spv: shaders/%.comp
	glslc $< -o $@

../zero_copy/shared_bo.o: ../zero_copy/shared_bo.cpp ../zero_copy/shared_bo.h
	$(MAKE) -C ../zero_copy libsharedbo.a

test_vk_attn_layer: test_vk_attn_layer.cpp gpu_attn_vk.cpp ../zero_copy/shared_bo.o $(SPVS)
	$(CXX) $(CXXFLAGS) test_vk_attn_layer.cpp gpu_attn_vk.cpp ../zero_copy/shared_bo.o -o $@ $(LDFLAGS) $(VK_LIBS)

clean:
	rm -f test_vk_attn_layer shaders/*.spv
