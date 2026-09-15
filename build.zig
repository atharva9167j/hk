const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const cuda_enabled = b.option(
        bool,
        "cuda",
        "Enable the CUDA GPU backend (dynamically loads CUDA driver API at runtime)",
    ) orelse true;

    const build_opts = b.addOptions();
    build_opts.addOption(bool, "cuda", cuda_enabled);
    const build_opts_mod = build_opts.createModule();

    // Core HK Module
    const hk_mod = b.addModule("hk", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_opts_mod },
        },
    });

    if (cuda_enabled) {
        if (target.result.os.tag != .windows) {
            hk_mod.linkSystemLibrary("dl", .{});
        }
    }

    // Shared Library (DLL / .so / .dylib) for C ABI bindings (Python ctypes, C#, etc.)
    // Reuses hk_mod directly (rather than recompiling src/root.zig into a second
    // module) so the CUDA object file / library links above apply here too.
    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "hk",
        .root_module = hk_mod,
    });
    b.installArtifact(lib);

    // HK CLI Tool
    const cli_mod = b.createModule(.{
        .root_source_file = b.path("tools/hk_cli.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "hk", .module = hk_mod },
        },
    });
    const cli_exe = b.addExecutable(.{
        .name = "hk",
        .root_module = cli_mod,
    });
    b.installArtifact(cli_exe);

    // CPU pp/tg throughput benchmark (no CUDA required)
    const cpu_bench_mod = b.createModule(.{
        .root_source_file = b.path("tools/hk_cpu_bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "hk", .module = hk_mod },
        },
    });
    const cpu_bench_exe = b.addExecutable(.{
        .name = "hk-cpu-bench",
        .root_module = cpu_bench_mod,
    });
    b.installArtifact(cpu_bench_exe);

    // Tests
    const tests_mod = b.createModule(.{
        .root_source_file = b.path("tests/roundtrip_tests.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "hk", .module = hk_mod },
        },
    });
    const tests = b.addTest(.{
        .root_module = tests_mod,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run HK library unit tests");
    test_step.dependOn(&run_tests.step);

    if (cuda_enabled) {
        const gpu_bench_mod = b.createModule(.{
            .root_source_file = b.path("tools/hk_gpu_bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "hk", .module = hk_mod },
            },
        });
        const gpu_bench_exe = b.addExecutable(.{
            .name = "hk-gpu-bench",
            .root_module = gpu_bench_mod,
        });
        b.installArtifact(gpu_bench_exe);

        const cuda_tests_mod = b.createModule(.{
            .root_source_file = b.path("tests/test_cuda.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "hk", .module = hk_mod },
            },
        });
        const cuda_tests = b.addTest(.{ .root_module = cuda_tests_mod });
        const run_cuda_tests = b.addRunArtifact(cuda_tests);
        const cuda_test_step = b.step("test-cuda", "Run CUDA GPU backend correctness tests");
        cuda_test_step.dependOn(&run_cuda_tests.step);
        test_step.dependOn(&run_cuda_tests.step);
    }
}
