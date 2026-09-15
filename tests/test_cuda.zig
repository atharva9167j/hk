const std = @import("std");
const hk = @import("hk");
const cuda = hk.cuda;
const tensor_ops = hk.tensor_ops;
const quantization = hk.quantization;

test "cuda device detected" {
    if (!cuda.isAvailable()) return error.SkipZigTest;
    try std.testing.expect(cuda.enabled);

    var name_buf: [256]u8 = undefined;
    _ = try cuda.getDeviceName(&name_buf);

    const total = try cuda.totalMemBytes();
    _ = try cuda.freeMemBytes();
    try std.testing.expect(total > 0);
}

test "cuda gemv f32 matches cpu reference" {
    if (!cuda.isAvailable()) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    const M: usize = 37;
    const K: usize = 129;

    const allocator = std.testing.allocator;

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
}

test "cuda gemv q8_0 matches cpu reference" {
    if (!cuda.isAvailable()) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    const M: usize = 11;
    const K: usize = 256;
    const blocks_per_row = K / 32;

    const allocator = std.testing.allocator;

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
}

test "cuda rmsnorm matches cpu reference" {
    if (!cuda.isAvailable()) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(101);
    const rand = prng.random();

    const dim: usize = 256;
    const allocator = std.testing.allocator;

    const x = try allocator.alloc(f32, dim);
    defer allocator.free(x);
    const w = try allocator.alloc(f32, dim);
    defer allocator.free(w);
    for (x) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
    for (w) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const out_cpu = try allocator.alloc(f32, dim);
    defer allocator.free(out_cpu);
    tensor_ops.rmsNormF32(x, w, 1e-5, out_cpu);

    const d_x = try cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(x));
    defer d_x.free();
    const d_w = try cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(w));
    defer d_w.free();
    const d_out = try cuda.DeviceBuffer.allocUninit(dim * @sizeOf(f32));
    defer d_out.free();

    try cuda.rmsNorm(d_x, d_w, d_out, dim, 1e-5);
    try cuda.synchronize();

    const out_gpu = try allocator.alloc(f32, dim);
    defer allocator.free(out_gpu);
    try d_out.download(std.mem.sliceAsBytes(out_gpu));

    for (out_cpu, out_gpu) |cv, gv| {
        try std.testing.expectApproxEqAbs(cv, gv, 1e-3);
    }
}

test "cuda swiglu matches reference" {
    if (!cuda.isAvailable()) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(202);
    const rand = prng.random();

    const hidden_dim: usize = 128;
    const allocator = std.testing.allocator;

    const gate = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(up);
    const ref = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(ref);

    for (0..hidden_dim) |i| {
        gate[i] = rand.float(f32) * 2.0 - 1.0;
        up[i] = rand.float(f32) * 2.0 - 1.0;
        const g = gate[i];
        const silu = g / (1.0 + @exp(-g));
        ref[i] = silu * up[i];
    }

    const d_gate = try cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(gate));
    defer d_gate.free();
    const d_up = try cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(up));
    defer d_up.free();

    try cuda.swiglu(d_gate, d_up, hidden_dim);
    try cuda.synchronize();

    const out_gpu = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(out_gpu);
    try d_gate.download(std.mem.sliceAsBytes(out_gpu));

    for (ref, out_gpu) |rv, gv| {
        try std.testing.expectApproxEqAbs(rv, gv, 1e-4);
    }
}

test "cuda add residual matches reference" {
    if (!cuda.isAvailable()) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(303);
    const rand = prng.random();

    const dim: usize = 256;
    const allocator = std.testing.allocator;

    const x = try allocator.alloc(f32, dim);
    defer allocator.free(x);
    const res = try allocator.alloc(f32, dim);
    defer allocator.free(res);
    const ref = try allocator.alloc(f32, dim);
    defer allocator.free(ref);

    for (0..dim) |i| {
        x[i] = rand.float(f32) * 2.0 - 1.0;
        res[i] = rand.float(f32) * 2.0 - 1.0;
        ref[i] = x[i] + res[i];
    }

    const d_x = try cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(x));
    defer d_x.free();
    const d_res = try cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(res));
    defer d_res.free();

    try cuda.addResidual(d_x, d_res, dim);
    try cuda.synchronize();

    const out_gpu = try allocator.alloc(f32, dim);
    defer allocator.free(out_gpu);
    try d_x.download(std.mem.sliceAsBytes(out_gpu));

    for (ref, out_gpu) |rv, gv| {
        try std.testing.expectApproxEqAbs(rv, gv, 1e-5);
    }
}

test "cuda stream and event profiling" {
    if (!cuda.isAvailable()) return error.SkipZigTest;

    var stream = try cuda.CudaStream.create();
    defer stream.destroy();

    var start = try cuda.CudaEvent.create();
    defer start.destroy();
    var end = try cuda.CudaEvent.create();
    defer end.destroy();

    try start.record(stream);

    const dim: usize = 1024;
    const allocator = std.testing.allocator;

    const buf = try allocator.alloc(f32, dim);
    defer allocator.free(buf);
    @memset(buf, 1.0);

    const d_buf = try cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(buf));
    defer d_buf.free();

    try cuda.swiglu(d_buf, d_buf, dim);
    try end.record(stream);

    try stream.synchronize();
    try end.synchronize();

    const elapsed = try cuda.CudaEvent.elapsedMs(&start, &end);
    try std.testing.expect(elapsed >= 0.0);
}
