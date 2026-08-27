# 1bit-monster.spec — RPM package for the 1bit.MONSTER inference engine
#
# Packages the prebuilt release binary + libraries from the staged tree
# produced by packaging/Makefile's `stage` target (the same artifacts the
# .deb and AppImage are built from). No source build happens in the spec —
# see packaging/Makefile's `package-rpm` target for how the staged tree is
# turned into the %{name}-%{version} source tarball rpmbuild consumes.
#
# Build:  make package-rpm   (from packaging/)
# Needs:  rpmbuild (Fedora: dnf install rpm-build; Ubuntu: apt install rpm)
#
# Runtime requirements NOT declared here (no packaging exists in any distro):
#   - TheRock (AMD ROCm 10.x) runtime for GPU/NPU acceleration — install to
#     /opt/rocm-therock and add its lib dirs to ld.so.conf, exactly as the
#     appliance ISO does (packaging/iso). Without it the engine runs CPU-only.
#   - XRT (libxrt2) + libwebsockets for the NPU engine sidecar and the
#     unified server — ordinary distro packages (Fedora: xrt, libwebsockets).

Name:          1bit-monster
Version:       2026.08.04
Release:       1%{?dist}
Summary:       One binary, all backends — NPU + GPU + CPU inference engine

License:       MIT
URL:           https://1bit.monster
Source0:       %{name}-%{version}.tar.gz

BuildArch:     x86_64
Requires:      glibc, libstdc++, libgomp

%description
1bit.MONSTER is a pure C++23 inference engine for AMD Strix Halo (Ryzen AI
Max+ 395): a single binary drives the XDNA 2 NPU, the Radeon 8060S GPU
(ROCm HIP + Vulkan), and CPU fallback in one process. 40 models across 16
families in the native 1BP format, plus direct GGUF execution. Zero Python,
zero Docker at runtime.

The single `1bit` ELF holds every server + CLI, dispatched by subcommand:
`zaya` (HIP inference + OpenAI-compatible API), `unified` (multi-backend
server), `router`, `jarvis`, `vision`, `chat`. Legacy names (zaya_server,
unified_server, …) are symlinks to it.

Requires a TheRock-compatible HIP runtime for GPU/NPU acceleration; CPU-only
operation works without it.

%prep
%setup -q -n %{name}-%{version}

%install
rm -rf %{buildroot}
# one binary + legacy-name symlinks
install -D -m 0755 usr/bin/1bit %{buildroot}%{_bindir}/1bit
for s in zaya_server unified_server unified_router vision_server jarvis_server onebitd onebit 1bit-server; do
    ln -s 1bit %{buildroot}%{_bindir}/$s
done
# shared HIP kernel library (dlopened as "librocm_cpp.so") — %{_libdir}
install -D -m 0755 usr/lib/x86_64-linux-gnu/librocm_cpp.so %{buildroot}%{_libdir}/librocm_cpp.so
# static helper libraries
for f in libbackend_manager.a libgguf_reader.a libvl_image.a; do
    install -D -m 0644 usr/lib/1bit/$f %{buildroot}%{_libdir}/1bit/$f
done
# video-lora + ZINC Vulkan shaders (present when the build produced them)
mkdir -p %{buildroot}%{_datadir}/1bit %{buildroot}%{_datadir}/1bit-monster
[ -d usr/share/1bit ] && cp -a usr/share/1bit/. %{buildroot}%{_datadir}/1bit/
[ -d usr/share/1bit-monster ] && cp -a usr/share/1bit-monster/. %{buildroot}%{_datadir}/1bit-monster/

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%files
%{_bindir}/1bit
%{_bindir}/zaya_server
%{_bindir}/unified_server
%{_bindir}/unified_router
%{_bindir}/vision_server
%{_bindir}/jarvis_server
%{_bindir}/onebitd
%{_bindir}/onebit
%{_bindir}/1bit-server
%{_libdir}/librocm_cpp.so
%{_libdir}/1bit/libbackend_manager.a
%{_libdir}/1bit/libgguf_reader.a
%{_libdir}/1bit/libvl_image.a
%{_datadir}/1bit
%{_datadir}/1bit-monster

%changelog
* Thu Aug 27 2026 1bit.MONSTER <admin@1bit.monster> - 2026.08.04-1
- Initial RPM packaging of the prebuilt 1bit.MONSTER engine.
