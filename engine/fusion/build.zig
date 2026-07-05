//! Build configuration for the fused NPU+GPU inference engine.
//! Links the NPU backend (XRT xclbin kernels), the unified scheduler, and
//! the GPU dispatch logic into a single binary.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });
    const target = b.standardTargetOptions(.{ .default_target = .{
        .cpu_arch = .x86_64,
        .os_tag = .linux,
    } });

    // ── XRT paths (NPU backend) ──
    const xrt_include = b.option([]const u8, "xrt-include", "XRT include dir") orelse
        "/home/bcloud/torch2aie/toolchain/xrt/include";
    const xrt_lib = b.option([]const u8, "xrt-lib", "XRT lib dir") orelse
        "/home/bcloud/torch2aie/toolchain/xrt/lib";

    // ── Unified scheduler module (shared KV cache) ──
    const sched_module = b.createModule(.{
        .root_source_file = b.path("../gpu/src/scheduler/scheduler.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ── NPU engine module (needs "sched" named import) ──
    const npu_engine_module = b.createModule(.{
        .root_source_file = b.path("../npu/src/npu_engine.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    npu_engine_module.addImport("sched", sched_module);

    // ── Fusion engine main module ──
    const root_mod = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Named imports for fusion modules
    root_mod.addImport("sched", sched_module);
    root_mod.addImport("npu_engine", npu_engine_module);

    // ── Executable ──
    const exe = b.addExecutable(.{
        .name = "fused-engine",
        .root_module = root_mod,
    });

    // NPU: C source for dequant
    root_mod.addCSourceFile(.{
        .file = b.path("../npu/src/dequant_q4nx.c"),
        .flags = &.{"-std=c11"},
    });

    // NPU: XRT libraries
    inline for (.{ "xrt_coreutil", "uuid", "m", "dl", "pthread" }) |lib| {
        root_mod.linkSystemLibrary(lib, .{});
    }
    root_mod.addIncludePath(.{ .cwd_relative = xrt_include });
    root_mod.addLibraryPath(.{ .cwd_relative = xrt_lib });
    root_mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });

    // ── Install ──
    b.installArtifact(exe);

    // ── Run step ──
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the fused NPU+GPU engine");
    run_step.dependOn(&run_cmd.step);

    // ── Test step ──
    const test_step = b.step("test", "Run unit tests");

    const test_sources = [_][]const u8{
        "dispatcher.zig",
        "memory.zig",
        // engine.zig is excluded from test — requires XRT hardware and
        // imports outside the module path (npu/, gpu/). Compile-tested
        // as part of the exe build.
    };

    for (test_sources) |src| {
        const test_mod = b.createModule(.{
            .root_source_file = b.path(src),
            .target = target,
            .optimize = optimize,
        });

        const test_exe = b.addTest(.{ .root_module = test_mod });
        const run_test = b.addRunArtifact(test_exe);
        test_step.dependOn(&run_test.step);
    }
}
