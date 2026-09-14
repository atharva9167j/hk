const std = @import("std");
const hk = @import("hk");
const cuda = hk.cuda;
const tensor_ops = hk.tensor_ops;
const quantization = hk.quantization;

test "cuda device detected" {
    try std.testing.expect(cuda.enabled);
    try std.testing.expect(cuda.isAvailable());

    var name_buf: [256]u8 = undefined;
    const name = try cuda.getDeviceName(&name_buf);
    std.debug.print("\n[cuda] device: {s}\n", .{name});

    const total = try cuda.totalMemBytes();
    const free = try cuda.freeMemBytes();
    std.debug.print("[cuda] total mem: {d} MiB, free: {d} MiB\n", .{
        @divTrunc(total, 1024 * 1024),
        @divTrunc(free, 1024 * 1024),
    });
    try std.testing.expect(total > 0);
}

test "cuda gemv f32 matches cpu reference" {
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    const M: usize = 37; // deliberately not a round number
    const K: usize = 129;

    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const W = try allocator.alloc(f32, M * K);
    defer allocator.free(W);
    const x = try allocator.alloc(f32, K);
    defer allocator.free(x);
    for (W) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
    for (x) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const y_cpu = try allocator.alloc(f32, M);
    defer allocator.free(y_cpu);
    tensor_ops.gemvF32(W, x, null, y_cpu, M, K);

    const d_w = try cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(W));
    defer d_w.free();
    const d_x = try cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(x));
    defer d_x.free();
    const d_y = try cuda.DeviceBuffer.allocUninit(M * @sizeOf(f32));
    defer d_y.free();

    try cuda.gemvF32(d_w, d_x, d_y, M, K);
    try cuda.synchronize();

    const y_gpu = try allocator.alloc(f32, M);
    defer allocator.free(y_gpu);
    try d_y.download(std.mem.sliceAsBytes(y_gpu));

    for (y_cpu, y_gpu) |cv, gv| {
        try std.testing.expectApproxEqAbs(cv, gv, 1e-2);
    }
    std.debug.print("[cuda] gemv f32: M={d} K={d} max_row_val_cpu={d:.4} gpu={d:.4} -- OK\n", .{ M, K, y_cpu[0], y_gpu[0] });
}

test "cuda gemv q8_0 matches cpu reference" {
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    const M: usize = 11;
    const K: usize = 256; // 8 blocks of 32
    const blocks_per_row = K / 32;

    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const W_f32 = try allocator.alloc(f32, M * K);
    defer allocator.free(W_f32);
    for (W_f32) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const W_q8 = try allocator.alloc(u8, M * blocks_per_row * 34);
    defer allocator.free(W_q8);
    for (0..M) |r| {
        for (0..blocks_per_row) |b| {
            var block: quantization.BlockQ8_0 = undefined;
            quantization.quantizeBlockQ8_0(W_f32[r * K + b * 32 .. r * K + b * 32 + 32], &block);
            const block_bytes = std.mem.asBytes(&block);
            const dst_off = (r * blocks_per_row + b) * 34;
            @memcpy(W_q8[dst_off .. dst_off + 34], block_bytes);
        }
    }

    const x = try allocator.alloc(f32, K);
    defer allocator.free(x);
    for (x) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const y_cpu = try allocator.alloc(f32, M);
    defer allocator.free(y_cpu);
    tensor_ops.gemvQ8_0(W_q8, x, null, y_cpu, M, K);

    const d_w = try cuda.DeviceBuffer.upload(W_q8);
    defer d_w.free();
    const d_x = try cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(x));
    defer d_x.free();
    const d_y = try cuda.DeviceBuffer.allocUninit(M * @sizeOf(f32));
    defer d_y.free();

    try cuda.gemvQ8_0(d_w, d_x, d_y, M, K);
    try cuda.synchronize();

    const y_gpu = try allocator.alloc(f32, M);
    defer allocator.free(y_gpu);
    try d_y.download(std.mem.sliceAsBytes(y_gpu));

    for (y_cpu, y_gpu) |cv, gv| {
        try std.testing.expectApproxEqAbs(cv, gv, 0.05);
    }
    std.debug.print("[cuda] gemv q8_0: M={d} K={d} row0_cpu={d:.4} row0_gpu={d:.4} -- OK\n", .{ M, K, y_cpu[0], y_gpu[0] });
}
