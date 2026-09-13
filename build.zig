const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Core HK Module
    const hk_mod = b.addModule("hk", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Shared Library (DLL / .so / .dylib) for C ABI bindings (Python ctypes, C#, etc.)
    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "hk",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(lib);

    // HK CLI Tool
    const cli_exe = b.addExecutable(.{
        .name = "hk",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/hk_cli.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "hk", .module = hk_mod },
            },
        }),
    });
    b.installArtifact(cli_exe);

    // Tests
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/roundtrip_tests.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "hk", .module = hk_mod },
            },
        }),
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run HK library unit tests");
    test_step.dependOn(&run_tests.step);
}
