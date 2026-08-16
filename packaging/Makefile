# 1bit.systems — build & package. Version is read from the root VERSION file
# (single source of truth — see issue #117); do not hardcode it here.
# One binary. All backends. Zero Python at runtime.
VERSION := $(shell [ -f ../VERSION ] && tr -d '[:space:]' < ../VERSION || echo "0.0.0")
BUILD_DIR := ../build
STAGE := build/stage/1bit-systems-$(VERSION)
.PHONY: all stage package-deb package-tarball package-appimage clean

all: stage

# ── Stage the real release binaries into the package tree ──
stage:
	mkdir -p $(STAGE)/usr/share/1bit
	mkdir -p $(STAGE)/usr/bin $(STAGE)/usr/lib/1bit $(STAGE)/usr/lib/x86_64-linux-gnu $(STAGE)/usr/share/1bit
	# ONE BINARY: every server + the CLI live in a single ELF (build/1bit);
	# legacy names are symlinks — argv[0] dispatch inside the ELF.
	cp $(BUILD_DIR)/1bit $(STAGE)/usr/bin/1bit
	chmod +x $(STAGE)/usr/bin/1bit
	for name in zaya_server unified_server unified_router vision_server onebitd onebit 1bit-server; do \
		ln -sf 1bit $(STAGE)/usr/bin/$$name; \
	done
	# optional sidecars (NPU engine needs XRT; video-lora CLI is a dev tool)
	if [ -f engine/npu/build/npu_engine_universal ]; then \
		cp engine/npu/build/npu_engine_universal $(STAGE)/usr/bin/1bit-npu; \
	fi
	if [ -f $(BUILD_DIR)/tools/video-lora/vulkan/video_lora_vk_cli ]; then \
		cp $(BUILD_DIR)/tools/video-lora/vulkan/video_lora_vk_cli $(STAGE)/usr/bin/video_lora_vk_cli; \
	fi
	cp $(BUILD_DIR)/librocm_cpp.so      $(STAGE)/usr/lib/x86_64-linux-gnu/
	cp $(BUILD_DIR)/libbackend_manager.a $(STAGE)/usr/lib/1bit/
	cp $(BUILD_DIR)/libgguf_reader.a    $(STAGE)/usr/lib/1bit/
	cp $(BUILD_DIR)/libvl_image.a       $(STAGE)/usr/lib/1bit/
	cp -r $(BUILD_DIR)/tools/video-lora/vulkan/shaders $(STAGE)/usr/share/1bit/shaders 2>/dev/null || true
	cp $(BUILD_DIR)/tools/video-lora/vulkan/libvideo_lora_vk.a $(STAGE)/usr/lib/1bit/ 2>/dev/null || true
	@echo "staged $(VERSION) -> $(STAGE)"

package-deb: stage
	rm -rf build/stage/deb
	mkdir -p build/stage/deb
	cp -r ../packaging/deb/DEBIAN build/stage/deb/
	cp -r $(STAGE)/usr build/stage/deb/
	dpkg-deb --build --root-owner-group build/stage/deb build/1bit-systems_$(VERSION)_amd64.deb
	@echo "deb: build/1bit-systems_$(VERSION)_amd64.deb"

package-tarball: stage
	tar czf build/1bit-systems-$(VERSION)-linux-amd64.tar.gz -C build/stage 1bit-systems-$(VERSION)
	@echo "tarball: build/1bit-systems-$(VERSION)-linux-amd64.tar.gz"

package-appimage: stage
	mkdir -p $(STAGE)/usr/share/applications
	cp ../packaging/appimage/AppDir/1bit.desktop $(STAGE)/usr/share/applications/
	cp ../packaging/appimage/AppDir/1bit.svg $(STAGE)/usr/share/icons/
	mkdir -p $(STAGE)/usr/share/icons
	@echo "AppDir staged at $(STAGE) — run appimagetool to bundle"

clean:
	rm -rf build/stage build/1bit-systems_$(VERSION)_amd64.deb build/1bit-systems-$(VERSION)-linux-amd64.tar.gz
