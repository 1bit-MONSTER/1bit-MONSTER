# 1bit.MONSTER — build & package. Version is read from the root VERSION file
# (single source of truth — see issue #117); do not hardcode it here.
# One binary. All backends. Zero Python at runtime.
VERSION := $(shell [ -f ../VERSION ] && tr -d '[:space:]' < ../VERSION || echo "0.0.0")
BUILD_DIR := ../build
STAGE := build/stage/1bit-monster-$(VERSION)
# Where the website's downloadable packages live (deployed to 1bit.monster
# from the site/ dir by .github/workflows/deploy.yml).
SITE_DL := ../site/downloads
# Non-root rpmbuild needs a writable rpm db (Ubuntu's /var/lib/rpm is root-owned)
RPM_DB ?= $(HOME)/.rpmdb
.PHONY: all stage package-deb package-tarball package-appimage package-rpm package-site package-snap package-flatpak clean

all: stage

# ── Stage the real release binaries into the package tree ──
stage:
	# Start from a pristine tree every run: stage is pure generated output and
	# several targets (package-rpm, package-deb) re-invoke it, so an incremental
	# re-run could otherwise leave stale/corrupted subtrees behind (previously:
	# a second cp -r of the ZINC shaders nested them one level deeper).
	rm -rf $(STAGE)
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
	# Idempotent shader staging: cp -r SRC DEST nests SRC inside DEST when DEST
	# already exists (a second `make stage`, e.g. from package-rpm's dependency,
	# produced usr/share/1bit-monster/shaders/shaders/ and doubled the .spv files
	# in the RPM). Copy the CONTENTS into a pre-created dir instead.
	cp -r $(BUILD_DIR)/tools/video-lora/vulkan/shaders/. $(STAGE)/usr/share/1bit/shaders/ 2>/dev/null || true
	# ZINC Vulkan shaders (embed.spv, flash_attn.spv, rms_norm_mul.spv, …) —
	# the Vulkan backend resolves them at runtime from /usr/share/1bit-monster/shaders
	# (see src/backend_zinc.cpp zinc_resolve_shader_dir). Without them the Vulkan
	# backend fails shader load and any Vulkan-routed model can't run.
	mkdir -p $(STAGE)/usr/share/1bit-monster/shaders
	cp -r $(BUILD_DIR)/zinc_cpp_build/shaders/. $(STAGE)/usr/share/1bit-monster/shaders/ 2>/dev/null || true
	cp $(BUILD_DIR)/tools/video-lora/vulkan/libvideo_lora_vk.a $(STAGE)/usr/lib/1bit/ 2>/dev/null || true
	@echo "staged $(VERSION) -> $(STAGE)"

package-deb: stage
	rm -rf build/stage/deb
	mkdir -p build/stage/deb
	cp -r ../packaging/deb/DEBIAN build/stage/deb/
	cp -r $(STAGE)/usr build/stage/deb/
	dpkg-deb --build --root-owner-group build/stage/deb build/1bit-monster_$(VERSION)_amd64.deb
	@echo "deb: build/1bit-monster_$(VERSION)_amd64.deb"

package-tarball: stage
	tar czf build/1bit-monster-$(VERSION)-linux-amd64.tar.gz -C build/stage 1bit-monster-$(VERSION)
	@echo "tarball: build/1bit-monster-$(VERSION)-linux-amd64.tar.gz"

package-appimage: stage
	mkdir -p $(STAGE)/usr/share/applications $(STAGE)/usr/share/icons
	cp ../packaging/appimage/AppDir/1bit.desktop $(STAGE)/usr/share/applications/
	cp ../packaging/appimage/AppDir/1bit.svg $(STAGE)/usr/share/icons/
	@echo "AppDir staged at $(STAGE) — run appimagetool to bundle"

package-rpm: stage
	# .rpm from the same staged tree as the .deb — see packaging/rpm/1bit-monster.spec
	rm -rf build/rpmbuild
	mkdir -p build/rpmbuild/BUILD build/rpmbuild/RPMS build/rpmbuild/SOURCES build/rpmbuild/SPECS build/rpmbuild/SRPMS
	rpm --initdb --dbpath $(RPM_DB) 2>/dev/null || true
	tar czf build/rpmbuild/SOURCES/1bit-monster-$(VERSION).tar.gz -C build/stage 1bit-monster-$(VERSION)
	sed 's|^Version: .*|Version: $(VERSION)|' ../packaging/rpm/1bit-monster.spec > build/rpmbuild/SPECS/1bit-monster.spec
	rpmbuild --define "_topdir $(abspath build/rpmbuild)" --define "_dbpath $(RPM_DB)" -bb build/rpmbuild/SPECS/1bit-monster.spec
	@echo "rpm: build/rpmbuild/RPMS/x86_64/1bit-monster-$(VERSION)-1.*.rpm"

package-site: stage
	# Sync the website's downloadable packages into ../site/downloads (deployed
	# to 1bit.monster by deploy.yml). The tarball is xz, NOT gzip: Cloudflare
	# Pages caps a single site asset at 25 MiB and the gzip tarball (~26 MB)
	# exceeds it — xz lands ~12 MB. Also regenerates SHA256SUMS + manifest.json
	# (read client-side by site/1bit-downloads.html).
	rm -f $(SITE_DL)/1bit-monster_*_amd64.deb $(SITE_DL)/1bit-monster-*-x86_64.AppImage $(SITE_DL)/1bit-monster-*-linux-amd64.tar.xz
	mkdir -p $(SITE_DL)
	cp build/1bit-monster_$(VERSION)_amd64.deb $(SITE_DL)/
	cp build/1bit-monster-$(VERSION)-x86_64.AppImage $(SITE_DL)/ 2>/dev/null || echo "warn: no AppImage in build/ (package-appimage + appimagetool) — manifest will skip it"
	tar -cJf $(SITE_DL)/1bit-monster-$(VERSION)-linux-amd64.tar.xz -C build/stage 1bit-monster-$(VERSION)
	# one-line installer, served from the website
	cp ../install.sh $(SITE_DL)/../install.sh
	python3 ../packaging/site-manifest.py "$(VERSION)" "$(abspath $(SITE_DL))"
	@echo "site packages -> $(SITE_DL)"

package-flatpak:
	bash ../packaging/flatpak/build-flatpak.sh

package-snap:
	# snapcraft pulls the pinned source-tag itself (packaging/snap/snapcraft.yaml);
	# needs snapcraft (Ubuntu: snap install snapcraft; others: see snapcraft.io)
	cd ../packaging/snap && snapcraft

clean:
	rm -rf build/stage build/1bit-monster_$(VERSION)_amd64.deb build/1bit-monster-$(VERSION)-linux-amd64.tar.gz build/rpmbuild
