const std = @import("std");
const format = @import("format.zig");
const nf4 = @import("nf4.zig");
const quantization = @import("quantization.zig");

pub const Vec4f = @Vector(4, f32);
pub const Vec8f = @Vector(8, f32);

const MAX_WORKERS: usize = 16;

pub fn getOptimalThreads(num_items: usize) usize {
    _ = num_items;
    return AtomicPool.get().total_threads;
}

pub const AtomicPool = struct {
    const TaskFn = *const fn (ctx: *anyopaque, start_r: usize, end_r: usize) void;

    const Worker = struct {
        work_seq: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
        done_seq: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
        _pad: [120]u8 = [_]u8{0} ** 120,
    };

    task_fn: ?TaskFn = null,
    task_ctx: ?*anyopaque = null,
    total_items: usize = 0,
    total_threads: usize = 0,
    seq: u32 = 0,
    shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    workers: [16]Worker = [_]Worker{.{}} ** 16,
    threads: [16]std.Thread = undefined,
    bg_workers: usize = 0,
    initialized: bool = false,

    pub fn get() *AtomicPool {
        const S = struct {
            var instance: AtomicPool = .{};
        };
        if (!S.instance.initialized) {
            S.instance.init();
        }
        return &S.instance;
    }

    pub fn init(self: *AtomicPool) void {
        if (self.initialized) return;
        const c = std.Thread.getCpuCount() catch 4;
        const total = std.math.clamp(c, 2, 8);
        self.bg_workers = total - 1;
        self.total_threads = total;

        for (0..self.bg_workers) |i| {
            self.threads[i] = std.Thread.spawn(.{}, workerLoop, .{ self, i + 1 }) catch {
                self.bg_workers = i;
                self.total_threads = i + 1;
                break;
            };
        }
        self.initialized = true;
    }

    fn workerLoop(self: *AtomicPool, worker_id: usize) void {
        var expected_seq: u32 = 1;
        while (!self.shutdown.load(.acquire)) {
            var spin: u32 = 0;
            while (self.workers[worker_id].work_seq.load(.acquire) != expected_seq) {
                if (self.shutdown.load(.acquire)) return;
                std.atomic.spinLoopHint();
                spin += 1;
                if (spin > 50_000) {
                    std.Thread.yield() catch {};
                    spin = 0;
                }
            }

            const total = self.total_items;
            const count = self.total_threads;
            const chunk = (total + count - 1) / count;
            const start = worker_id * chunk;
            const end = @min(start + chunk, total);

            if (start < total) {
                self.task_fn.?(self.task_ctx.?, start, end);
            }

            self.workers[worker_id].done_seq.store(expected_seq, .release);
            expected_seq +%= 1;
        }
    }

    pub fn parallelFor(self: *AtomicPool, total: usize, ctx: *anyopaque, task: TaskFn) void {
        if (self.total_threads <= 1 or total < 64) {
            task(ctx, 0, total);
            return;
        }

        const count = self.total_threads;
        self.total_items = total;
        self.task_ctx = ctx;
        self.task_fn = task;

        self.seq +%= 1;
        const cur_seq = self.seq;

        for (1..count) |i| {
            self.workers[i].work_seq.store(cur_seq, .release);
        }

        const chunk = (total + count - 1) / count;
        const end0 = @min(chunk, total);
        task(ctx, 0, end0);

        for (1..count) |i| {
            var spin: u32 = 0;
            while (self.workers[i].done_seq.load(.acquire) != cur_seq) {
                std.atomic.spinLoopHint();
                spin += 1;
                if (spin > 50_000) {
                    std.Thread.yield() catch {};
                    spin = 0;
                }
            }
        }
    }
};

/// Fast SIMD dot product of two f32 slices using 4 independent accumulators
pub fn dotProductF32(a: []const f32, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    var i: usize = 0;

    var acc0: Vec8f = @splat(0.0);
    var acc1: Vec8f = @splat(0.0);
    var acc2: Vec8f = @splat(0.0);
    var acc3: Vec8f = @splat(0.0);

    // 32-wide SIMD unrolled loop with 4 accumulators (saturates dual FMA pipelines)
    while (i + 32 <= len) : (i += 32) {
        const va0: Vec8f = a[i + 0 ..][0..8].*;
        const vb0: Vec8f = b[i + 0 ..][0..8].*;
        acc0 += va0 * vb0;

        const va1: Vec8f = a[i + 8 ..][0..8].*;
        const vb1: Vec8f = b[i + 8 ..][0..8].*;
        acc1 += va1 * vb1;

        const va2: Vec8f = a[i + 16 ..][0..8].*;
        const vb2: Vec8f = b[i + 16 ..][0..8].*;
        acc2 += va2 * vb2;

        const va3: Vec8f = a[i + 24 ..][0..8].*;
        const vb3: Vec8f = b[i + 24 ..][0..8].*;
        acc3 += va3 * vb3;
    }

    var acc = (acc0 + acc1) + (acc2 + acc3);

    // 8-wide SIMD loop
    while (i + 8 <= len) : (i += 8) {
        const va: Vec8f = a[i..][0..8].*;
        const vb: Vec8f = b[i..][0..8].*;
        acc += va * vb;
    }

    var sum: f32 = @reduce(.Add, acc);

    // 4-wide SIMD loop
    while (i + 4 <= len) : (i += 4) {
        const va: Vec4f = a[i..][0..4].*;
        const vb: Vec4f = b[i..][0..4].*;
        sum += @reduce(.Add, va * vb);
    }

    // Scalar remainder
    while (i < len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

const GemvF32Ctx = struct {
    W: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    K: usize,
};

fn gemvF32Task(ctx_ptr: *anyopaque, start_r: usize, end_r: usize) void {
    const ctx: *const GemvF32Ctx = @ptrCast(@alignCast(ctx_ptr));
    for (start_r..end_r) |r| {
        const row = ctx.W[r * ctx.K .. (r + 1) * ctx.K];
        var dot = dotProductF32(row, ctx.x);
        if (ctx.bias) |b| {
            if (r < b.len) dot += b[r];
        }
        ctx.y[r] = dot;
    }
}

/// Fast Multi-Threaded Matrix-Vector Multiplication: y = W * x + (bias)
/// W is shape [M, K] in row-major order.
pub fn gemvF32(
    W: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
) void {
    const pool = AtomicPool.get();
    var ctx = GemvF32Ctx{
        .W = W,
        .x = x,
        .bias = bias,
        .y = y,
        .K = K,
    };
    pool.parallelFor(M, @ptrCast(&ctx), gemvF32Task);
}

/// Fast Matrix Multiplication: C = A * B
/// A is [M, K], B is [K, N], C is [M, N] (row-major)
pub fn gemmF32(
    A: []const f32,
    B: []const f32,
    C: []f32,
    M: usize,
    K: usize,
    N: usize,
) void {
    @memset(C, 0.0);
    for (0..M) |m| {
        for (0..K) |k| {
            const a_val = A[m * K + k];
            const b_row = B[k * N .. (k + 1) * N];
            const c_row = C[m * N .. (m + 1) * N];

            var n: usize = 0;
            const va: Vec8f = @splat(a_val);
            while (n + 8 <= N) : (n += 8) {
                const vb: Vec8f = b_row[n..][0..8].*;
                var vc: Vec8f = c_row[n..][0..8].*;
                vc += va * vb;
                c_row[n..][0..8].* = vc;
            }
            while (n < N) : (n += 1) {
                C[m * N + n] += a_val * B[k * N + n];
            }
        }
    }
}

fn fusedGemvNF4Worker(
    packed_W: []const u8,
    scales: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    start_r: usize,
    end_r: usize,
    K: usize,
    block_size: usize,
    blocks_per_row: usize,
) void {
    const k_bytes = (K + 1) / 2;
    for (start_r..end_r) |r| {
        const row_packed = packed_W[r * k_bytes .. (r + 1) * k_bytes];
        var dot: f32 = 0.0;
        var global_block_idx = r * blocks_per_row;

        var k: usize = 0;
        while (k < K) {
            const block_scale = if (global_block_idx < scales.len) scales[global_block_idx] else 1.0;
            global_block_idx += 1;

            const cur_block_len = @min(block_size, K - k);
            for (0..cur_block_len) |bi| {
                const elem_idx = k + bi;
                const byte_val = row_packed[elem_idx / 2];
                const code: usize = if (elem_idx % 2 == 0)
                    (byte_val & 0x0F)
                else
                    ((byte_val >> 4) & 0x0F);

                const w_val = nf4.NF4_TABLE[code] * block_scale;
                dot += w_val * x[elem_idx];
            }
            k += cur_block_len;
        }

        if (bias) |b| {
            if (r < b.len) dot += b[r];
        }
        y[r] = dot;
    }
}

const FusedGemvNF4Ctx = struct {
    packed_W: []const u8,
    scales: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    K: usize,
    block_size: usize,
    blocks_per_row: usize,
};

fn fusedGemvNF4Task(ctx_ptr: *anyopaque, start_r: usize, end_r: usize) void {
    const ctx: *const FusedGemvNF4Ctx = @ptrCast(@alignCast(ctx_ptr));
    fusedGemvNF4Worker(ctx.packed_W, ctx.scales, ctx.x, ctx.bias, ctx.y, start_r, end_r, ctx.K, ctx.block_size, ctx.blocks_per_row);
}

/// Fused NF4 Dequantize-and-GEMV: computes y = Dequant(W_nf4) * x + bias
/// Does not materialize dequantized float32 weight matrix in RAM!
pub fn fusedGemvNF4(
    packed_W: []const u8,
    scales: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
    block_size: usize,
) void {
    const blocks_per_row = (K + block_size - 1) / block_size;
    const pool = AtomicPool.get();
    var ctx = FusedGemvNF4Ctx{
        .packed_W = packed_W,
        .scales = scales,
        .x = x,
        .bias = bias,
        .y = y,
        .K = K,
        .block_size = block_size,
        .blocks_per_row = blocks_per_row,
    };
    pool.parallelFor(M, @ptrCast(&ctx), fusedGemvNF4Task);
}

fn fusedGemvDQ8Worker(
    W_i8: []const i8,
    scales: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    start_r: usize,
    end_r: usize,
    K: usize,
    block_size: usize,
    blocks_per_row: usize,
) void {
    for (start_r..end_r) |r| {
        const row_i8 = W_i8[r * K .. (r + 1) * K];
        var dot: f32 = 0.0;
        var global_block_idx = r * blocks_per_row;

        var k: usize = 0;
        while (k < K) {
            const block_scale = if (global_block_idx < scales.len) scales[global_block_idx] else 1.0;
            global_block_idx += 1;

            const cur_block_len = @min(block_size, K - k);
            var bi: usize = 0;
            while (bi + 8 <= cur_block_len) : (bi += 8) {
                const elem_idx = k + bi;
                const w_i8_vec: @Vector(8, i8) = row_i8[elem_idx..][0..8].*;
                const w_f32_vec: Vec8f = @floatFromInt(w_i8_vec);
                const x_vec: Vec8f = x[elem_idx..][0..8].*;
                dot += @reduce(.Add, w_f32_vec * x_vec) * block_scale;
            }
            while (bi < cur_block_len) : (bi += 1) {
                const elem_idx = k + bi;
                const w_val = @as(f32, @floatFromInt(row_i8[elem_idx])) * block_scale;
                dot += w_val * x[elem_idx];
            }
            k += cur_block_len;
        }

        if (bias) |b| {
            if (r < b.len) dot += b[r];
        }
        y[r] = dot;
    }
}

const FusedGemvDQ8Ctx = struct {
    W_i8: []const i8,
    scales: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    K: usize,
    block_size: usize,
    blocks_per_row: usize,
};

fn fusedGemvDQ8Task(ctx_ptr: *anyopaque, start_r: usize, end_r: usize) void {
    const ctx: *const FusedGemvDQ8Ctx = @ptrCast(@alignCast(ctx_ptr));
    fusedGemvDQ8Worker(ctx.W_i8, ctx.scales, ctx.x, ctx.bias, ctx.y, start_r, end_r, ctx.K, ctx.block_size, ctx.blocks_per_row);
}

/// Fused DQ8 Dequantize-and-GEMV: computes y = (W_i8 * scale) * x + bias
pub fn fusedGemvDQ8(
    W_i8: []const i8,
    scales: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
    block_size: usize,
) void {
    const blocks_per_row = (K + block_size - 1) / block_size;
    const pool = AtomicPool.get();
    var ctx = FusedGemvDQ8Ctx{
        .W_i8 = W_i8,
        .scales = scales,
        .x = x,
        .bias = bias,
        .y = y,
        .K = K,
        .block_size = block_size,
        .blocks_per_row = blocks_per_row,
    };
    pool.parallelFor(M, @ptrCast(&ctx), fusedGemvDQ8Task);
}

/// Native compiled Ampere 2:4 structured packing
/// Output layout: [meta: ceil(N/8) bytes] [pad to 4-byte boundary] [values: (N/2) * 4 bytes (f32)]
pub fn nativePack24(
    dense_in: []const f32,
    out_meta: []u8,
    out_values: []f32,
) usize {
    const num_groups = dense_in.len / 4;
    @memset(out_meta, 0);

    for (0..num_groups) |g| {
        const b = dense_in[g * 4 .. (g + 1) * 4];

        // Find top 2 indices by absolute magnitude
        var best0: usize = 0;
        var best1: usize = 1;
        var max0 = @abs(b[0]);
        var max1 = @abs(b[1]);

        if (max1 > max0) {
            std.mem.swap(usize, &best0, &best1);
            std.mem.swap(f32, &max0, &max1);
        }

        for (2..4) |i| {
            const mag = @abs(b[i]);
            if (mag > max0) {
                best1 = best0;
                max1 = max0;
                best0 = i;
                max0 = mag;
            } else if (mag > max1) {
                best1 = i;
                max1 = mag;
            }
        }

        // Sort indices so idx0 < idx1
        var idx0 = best0;
        var idx1 = best1;
        if (idx0 > idx1) std.mem.swap(usize, &idx0, &idx1);

        out_values[g * 2 + 0] = b[idx0];
        out_values[g * 2 + 1] = b[idx1];

        const nibble: u8 = @as(u8, @intCast((idx0 & 0x03) | ((idx1 & 0x03) << 2)));
        const meta_idx = g / 2;
        if (g % 2 == 0) {
            out_meta[meta_idx] |= (nibble & 0x0F);
        } else {
            out_meta[meta_idx] |= ((nibble & 0x0F) << 4);
        }
    }

    return num_groups * 2;
}

/// Native compiled Ampere 2:4 structured unpacking
pub fn nativeUnpack24(
    meta: []const u8,
    values: []const f32,
    out_dense: []f32,
) void {
    const num_groups = out_dense.len / 4;
    @memset(out_dense, 0.0);

    for (0..num_groups) |g| {
        const meta_byte = meta[g / 2];
        const nibble: u8 = if (g % 2 == 0) (meta_byte & 0x0F) else ((meta_byte >> 4) & 0x0F);

        const idx0: usize = nibble & 0x03;
        const idx1: usize = (nibble >> 2) & 0x03;

        const val0 = values[g * 2 + 0];
        const val1 = values[g * 2 + 1];

        out_dense[g * 4 + idx0] = val0;
        out_dense[g * 4 + idx1] = val1;
    }
}

/// Native SiLU (Swish) activation: f(x) = x / (1.0 + exp(-x))
pub fn siluF32(x: []const f32, out: []f32) void {
    const len = @min(x.len, out.len);
    for (0..len) |i| {
        const v = x[i];
        out[i] = v / (1.0 + @exp(-v));
    }
}

/// Native GELU activation: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn geluF32(x: []const f32, out: []f32) void {
    const len = @min(x.len, out.len);
    const sqrt_2_over_pi: f32 = 0.7978845608;
    for (0..len) |i| {
        const v = x[i];
        const inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
        // clamp to avoid tanh overflow
        const clamped = std.math.clamp(inner, -10.0, 10.0);
        const exp_2x = @exp(2.0 * clamped);
        const tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0);
        out[i] = 0.5 * v * (1.0 + tanh_val);
    }
}

/// Element-wise Hadamard product: out[i] = a[i] * b[i]
pub fn elementWiseMulF32(a: []const f32, b: []const f32, out: []f32) void {
    const len = @min(@min(a.len, b.len), out.len);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const va: Vec8f = a[i..][0..8].*;
        const vb: Vec8f = b[i..][0..8].*;
        out[i..][0..8].* = va * vb;
    }
    while (i < len) : (i += 1) {
        out[i] = a[i] * b[i];
    }
}

/// Fast RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight
pub fn rmsNormF32(
    x: []const f32,
    weight: []const f32,
    eps: f32,
    out: []f32,
) void {
    const len = @min(@min(x.len, weight.len), out.len);
    if (len == 0) return;

    var sum_sq: f32 = 0.0;
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const vx: Vec8f = x[i..][0..8].*;
        sum_sq += @reduce(.Add, vx * vx);
    }
    while (i < len) : (i += 1) {
        sum_sq += x[i] * x[i];
    }

    const mean_sq = sum_sq / @as(f32, @floatFromInt(len));
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);

    var j: usize = 0;
    const v_inv_rms: Vec8f = @splat(inv_rms);
    while (j + 8 <= len) : (j += 8) {
        const vx: Vec8f = x[j..][0..8].*;
        const vw: Vec8f = weight[j..][0..8].*;
        out[j..][0..8].* = vx * v_inv_rms * vw;
    }
    while (j < len) : (j += 1) {
        out[j] = x[j] * inv_rms * weight[j];
    }
}

/// Fast LayerNorm: y = ((x - mean) / sqrt(variance + eps)) * weight + (bias)
pub fn layerNormF32(
    x: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    eps: f32,
    out: []f32,
) void {
    const len = @min(@min(x.len, weight.len), out.len);
    if (len == 0) return;

    var sum: f32 = 0.0;
    for (0..len) |i| sum += x[i];
    const mean = sum / @as(f32, @floatFromInt(len));

    var var_sum: f32 = 0.0;
    for (0..len) |i| {
        const diff = x[i] - mean;
        var_sum += diff * diff;
    }
    const inv_std = 1.0 / @sqrt((var_sum / @as(f32, @floatFromInt(len))) + eps);

    for (0..len) |i| {
        var val = (x[i] - mean) * inv_std * weight[i];
        if (bias) |b| {
            if (i < b.len) val += b[i];
        }
        out[i] = val;
    }
}

/// Native Fused SwiGLU Forward Pass:
/// y = W_down * (SiLU(W_gate * x + b_gate) * (W_up * x + b_up)) + b_down
/// Computes without any external Python or PyTorch runtime.
pub fn forwardSwiGLUF32(
    x: []const f32, // [in_f]
    w_gate: []const f32, // [inter_f, in_f]
    b_gate: ?[]const f32, // [inter_f]
    w_up: []const f32, // [inter_f, in_f]
    b_up: ?[]const f32, // [inter_f]
    w_down: []const f32, // [out_f, inter_f]
    b_down: ?[]const f32, // [out_f]
    inter_buffer: []f32, // scratch buffer of at least 2 * inter_f
    out: []f32, // [out_f]
    in_f: usize,
    inter_f: usize,
    out_f: usize,
) !void {
    if (inter_buffer.len < inter_f * 2) return error.BufferTooSmall;
    if (out.len < out_f) return error.BufferTooSmall;

    const gate_buf = inter_buffer[0..inter_f];
    const up_buf = inter_buffer[inter_f .. inter_f * 2];

    // 1. gate = W_gate * x + b_gate
    gemvF32(w_gate, x, b_gate, gate_buf, inter_f, in_f);

    // 2. up = W_up * x + b_up
    gemvF32(w_up, x, b_up, up_buf, inter_f, in_f);

    // 3. act = SiLU(gate) * up
    for (0..inter_f) |i| {
        const g = gate_buf[i];
        const silu_g = g / (1.0 + @exp(-g));
        gate_buf[i] = silu_g * up_buf[i]; // in-place combined activation in gate_buf
    }

    // 4. out = W_down * act + b_down
    gemvF32(w_down, gate_buf, b_down, out, out_f, inter_f);
}

/// Numerically stable Softmax: p_i = exp(z_i - max) / sum(exp(z_j - max))
pub fn softmaxF32(logits: []const f32, out_probs: []f32) void {
    const len = @min(logits.len, out_probs.len);
    if (len == 0) return;

    var max_val: f32 = logits[0];
    for (1..len) |i| {
        if (logits[i] > max_val) max_val = logits[i];
    }

    var sum: f32 = 0.0;
    for (0..len) |i| {
        const e = @exp(logits[i] - max_val);
        out_probs[i] = e;
        sum += e;
    }

    const inv_sum = 1.0 / sum;
    for (0..len) |i| {
        out_probs[i] *= inv_sum;
    }
}

/// Fast Argmax
pub fn argmaxF32(logits: []const f32) usize {
    if (logits.len == 0) return 0;
    var best_idx: usize = 0;
    var best_val: f32 = logits[0];
    for (1..logits.len) |i| {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

/// Permutes weights from Hugging Face half-split RoPE layout to GGUF/llama.cpp interleaved layout.
pub fn ropePermuteHFToGGUF(in: []const f32, out: []f32, n_heads: usize, head_dim: usize) void {
    const head_size = n_heads * head_dim;
    if (head_size == 0) return;
    const n_batches = in.len / head_size;
    const half = head_dim / 2;

    for (0..n_batches) |b| {
        const batch_in = in[b * head_size .. (b + 1) * head_size];
        const batch_out = out[b * head_size .. (b + 1) * head_size];

        for (0..n_heads) |h| {
            const src_h = batch_in[h * head_dim .. (h + 1) * head_dim];
            const dst_h = batch_out[h * head_dim .. (h + 1) * head_dim];
            for (0..half) |i| {
                dst_h[2 * i] = src_h[i];
                dst_h[2 * i + 1] = src_h[half + i];
            }
        }
    }
}

/// Unpermutes weights from GGUF/llama.cpp interleaved layout to Hugging Face half-split layout.
pub fn ropeUnpermuteGGUFToHF(in: []const f32, out: []f32, n_heads: usize, head_dim: usize) void {
    const head_size = n_heads * head_dim;
    if (head_size == 0) return;
    const n_batches = in.len / head_size;
    const half = head_dim / 2;

    for (0..n_batches) |b| {
        const batch_in = in[b * head_size .. (b + 1) * head_size];
        const batch_out = out[b * head_size .. (b + 1) * head_size];

        for (0..n_heads) |h| {
            const src_h = batch_in[h * head_dim .. (h + 1) * head_dim];
            const dst_h = batch_out[h * head_dim .. (h + 1) * head_dim];
            for (0..half) |i| {
                dst_h[i] = src_h[2 * i];
                dst_h[half + i] = src_h[2 * i + 1];
            }
        }
    }
}

/// Applies an additive offset (e.g. +1.0 or -1.0 for Gemma/T5 LayerNorm/RMSNorm)
pub fn layerNormOffsetF32(data: []f32, offset: f32) void {
    var i: usize = 0;
    const v_off: Vec8f = @splat(offset);
    while (i + 8 <= data.len) : (i += 8) {
        var v: Vec8f = data[i..][0..8].*;
        v += v_off;
        data[i..][0..8].* = v;
    }
    while (i < data.len) : (i += 1) {
        data[i] += offset;
    }
}

pub fn gemvQ8_0RowBytes(W_row_bytes: []const u8, x: []const f32, blocks_per_row: usize) f32 {
    var total_sum: f32 = 0.0;
    for (0..blocks_per_row) |b| {
        const blk_offset = b * 34;
        const d_raw = std.mem.readInt(u16, W_row_bytes[blk_offset..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_raw)));
        const qs_bytes = W_row_bytes[blk_offset + 2 .. blk_offset + 34];
        const x_sub = x[b * 32 .. (b + 1) * 32];
        var block_sum: f32 = 0.0;
        for (0..32) |i| {
            const q: i8 = @bitCast(qs_bytes[i]);
            block_sum += @as(f32, @floatFromInt(q)) * x_sub[i];
        }
        total_sum += block_sum * d;
    }
    return total_sum;
}

pub fn gemvQ8_0Row(row_blocks: []const quantization.BlockQ8_0, x: []const f32) f32 {
    const bytes = std.mem.sliceAsBytes(row_blocks);
    return gemvQ8_0RowBytes(bytes, x, row_blocks.len);
}

/// Matrix-Vector Multiplication for Q8_0 quantized matrix: y = W_q8_0 * x + bias
/// W_bytes contains M * (K / 32) * 34 bytes.
/// x is length K, y is length M.
pub fn gemvQ8_0(
    W_bytes: []const u8,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
) void {
    const blocks_per_row = K / 32;
    const row_bytes_len = blocks_per_row * 34;
    for (0..M) |r| {
        const row_bytes = W_bytes[r * row_bytes_len .. (r + 1) * row_bytes_len];
        var row_sum = gemvQ8_0RowBytes(row_bytes, x, blocks_per_row);
        if (bias) |b| {
            if (r < b.len) row_sum += b[r];
        }
        y[r] = row_sum;
    }
}

pub fn gemvQ4_0RowBytes(W_row_bytes: []const u8, x: []const f32, blocks_per_row: usize) f32 {
    var total_sum: f32 = 0.0;
    for (0..blocks_per_row) |b| {
        const blk_offset = b * 18;
        const d_raw = std.mem.readInt(u16, W_row_bytes[blk_offset..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_raw)));
        const qs_bytes = W_row_bytes[blk_offset + 2 .. blk_offset + 18];
        const x_sub = x[b * 32 .. (b + 1) * 32];
        var block_sum: f32 = 0.0;
        for (0..16) |i| {
            const byte = qs_bytes[i];
            const q0: i8 = @as(i8, @intCast(byte & 0x0F)) - 8;
            const q1: i8 = @as(i8, @intCast((byte >> 4) & 0x0F)) - 8;
            block_sum += @as(f32, @floatFromInt(q0)) * x_sub[i] + @as(f32, @floatFromInt(q1)) * x_sub[i + 16];
        }
        total_sum += block_sum * d;
    }
    return total_sum;
}

pub fn gemvQ4_0Row(row_blocks: []const quantization.BlockQ4_0, x: []const f32) f32 {
    const bytes = std.mem.sliceAsBytes(row_blocks);
    return gemvQ4_0RowBytes(bytes, x, row_blocks.len);
}

/// Matrix-Vector Multiplication for Q4_0 quantized matrix: y = W_q4_0 * x + bias
/// W_bytes contains M * (K / 32) * 18 bytes.
pub fn gemvQ4_0(
    W_bytes: []const u8,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
) void {
    const blocks_per_row = K / 32;
    const row_bytes_len = blocks_per_row * 18;
    for (0..M) |r| {
        const row_bytes = W_bytes[r * row_bytes_len .. (r + 1) * row_bytes_len];
        var row_sum = gemvQ4_0RowBytes(row_bytes, x, blocks_per_row);
        if (bias) |b| {
            if (r < b.len) row_sum += b[r];
        }
        y[r] = row_sum;
    }
}

/// Matrix-Vector Multiplication for Q4_K quantized matrix: y = W_q4_k * x + bias
pub fn gemvQ4_K(
    W_bytes: []const u8,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
) void {
    const superblocks_per_row = K / 256;
    const row_bytes_len = superblocks_per_row * 144;
    var deq_buf: [256]f32 = undefined;
    var blk: quantization.BlockQ4_K = undefined;
    const blk_slice = std.mem.asBytes(&blk);

    for (0..M) |r| {
        var row_sum: f32 = 0.0;
        const row_bytes = W_bytes[r * row_bytes_len .. (r + 1) * row_bytes_len];
        for (0..superblocks_per_row) |sb_idx| {
            const sb_offset = sb_idx * 144;
            @memcpy(blk_slice, row_bytes[sb_offset .. sb_offset + 144]);
            quantization.dequantizeSuperBlockQ4_K(&blk, 256, &deq_buf);
            const x_sub = x[sb_idx * 256 .. (sb_idx + 1) * 256];
            row_sum += dotProductF32(&deq_buf, x_sub);
        }
        if (bias) |b| {
            if (r < b.len) row_sum += b[r];
        }
        y[r] = row_sum;
    }
}


