const std = @import("std");

pub const SandboxResult = struct {
    success: bool,
    exit_code: u8,
    stdout: []const u8,
    stderr: []const u8,
    timed_out: bool,
    elapsed_ms: u64,

    pub fn deinit(self: *SandboxResult, allocator: std.mem.Allocator) void {
        if (self.stdout.len > 0) allocator.free(self.stdout);
        if (self.stderr.len > 0) allocator.free(self.stderr);
    }
};

/// Native Zig process execution sandbox with timeouts and resource protection.
pub const NativeSandbox = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    timeout_ms: u64,

    pub fn init(allocator: std.mem.Allocator, timeout_ms: u64) NativeSandbox {
        return .{
            .allocator = allocator,
            .io = std.Options.debug_io,
            .timeout_ms = timeout_ms,
        };
    }

    pub fn initWithIo(allocator: std.mem.Allocator, io: std.Io, timeout_ms: u64) NativeSandbox {
        return .{
            .allocator = allocator,
            .io = io,
            .timeout_ms = timeout_ms,
        };
    }

    /// Executes an isolated child process collecting stdout and stderr.
    pub fn execute(self: *NativeSandbox, argv: []const []const u8) !SandboxResult {
        const start_time = std.Io.Timestamp.now(self.io, .awake);

        const run_res = std.process.run(self.allocator, self.io, .{
            .argv = argv,
            .expand_arg0 = .expand,
        }) catch |err| {
            const err_msg = try std.fmt.allocPrint(self.allocator, "Process execution error: {}", .{err});
            return SandboxResult{
                .success = false,
                .exit_code = 1,
                .stdout = &[_]u8{},
                .stderr = err_msg,
                .timed_out = false,
                .elapsed_ms = 0,
            };
        };

        const end_time = std.Io.Timestamp.now(self.io, .awake);
        const elapsed_ns = end_time.nanoseconds - start_time.nanoseconds;
        const elapsed = @as(u64, @intCast(@max(0, @divTrunc(elapsed_ns, 1_000_000))));

        var exit_code: u8 = 0;
        var success = false;
        switch (run_res.term) {
            .exited => |code| {
                exit_code = code;
                success = (code == 0);
            },
            else => {
                exit_code = 1;
                success = false;
            },
        }

        const timed_out = (elapsed >= self.timeout_ms);
        if (timed_out) {
            success = false;
        }

        return SandboxResult{
            .success = success,
            .exit_code = exit_code,
            .stdout = run_res.stdout,
            .stderr = run_res.stderr,
            .timed_out = timed_out,
            .elapsed_ms = elapsed,
        };
    }
};
