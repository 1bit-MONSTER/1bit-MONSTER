const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const compile_shaders = b.option(bool, "shaders", "Compile GLSL shaders to SPIR-V (requires glslc)") orelse (builtin.os.tag == .linux);

    // ---- Shaders -------------------------------------------------------
    if (compile_shaders) {
        const shader_dir = "shaders";
        const shader_names: []const []const u8 = &.{
            "conv2d", "group_norm", "silu", "elementwise", "attention", "lora_merge",
        };

        for (shader_names) |name| {
            const comp_file = b.pathJoin(&.{ shader_dir, b.fmt("{s}.comp", .{name}) });
            const spv_file = b.pathJoin(&.{ shader_dir, b.fmt("{s}.spv", .{name}) });

            const compile_step = b.addSystemCommand(&.{
                "glslc", "-o", b.pathJoin(&.{ b.install_path, spv_file }),
                comp_file,
            });
            b.default_step.dependOn(&compile_step.step);
        }
    }

    // ---- Executable ----------------------------------------------------
    const exe = b.addExecutable(.{
        .name = "video-lora-vk",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.linkSystemLibrary("vulkan");
    exe.linkLibC();
    b.installArtifact(exe);

    // ---- Tests ---------------------------------------------------------
    const test_step = b.step("test", "Run unit tests");
    const tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    tests.linkSystemLibrary("vulkan");
    tests.linkLibC();
    const run_tests = b.addRunArtifact(tests);
    test_step.dependOn(&run_tests.step);
}
