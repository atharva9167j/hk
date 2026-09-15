const std = @import("std");
const format = @import("format.zig");
const nf4 = @import("nf4.zig");
const quantization = @import("quantization.zig");

pub const Vec4f = @Vector(4, f32);
pub const Vec8f = @Vector(8, f32);

const MAX_WORKERS: usize = 16;

pub fn getOptimalThreads(num_items: usize) usize {
    _ = num_items;
    return 1;
}

pub const AtomicPool = struct {
    const TaskFn = *const fn (ctx: *anyopaque, start_r: usize, end_r: usize) void;
    total_threads: usize = 1,

    pub fn get() *AtomicPool {
        const S = struct {
            var instance: AtomicPool = .{};
        };
        return &S.instance;
    }

    pub fn parallelFor(self: *AtomicPool, total: usize, ctx: *anyopaque, task: TaskFn) void {
        _ = self;
        if (total > 0) {
            task(ctx, 0, total);
        }
    }
};

/// Fast SIMD dot product of two f32 slices using 4 independent 8-wide vector accumulators (32 floats/iter)
pub fn dotProductF32(a: []const f32, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    var i: usize = 0;

    var acc0: @Vector(8, f32) = @splat(0.0);
    var acc1: @Vector(8, f32) = @splat(0.0);
    var acc2: @Vector(8, f32) = @splat(0.0);
    var acc3: @Vector(8, f32) = @splat(0.0);

    // 4 independent vector accumulators unrolled (32 elements per iteration)
    // Saturates dual 256-bit AVX2/AVX-512 FMA execution units
    while (i + 32 <= len) : (i += 32) {
        const va0: @Vector(8, f32) = a[i + 0 ..][0..8].*;
        const vb0: @Vector(8, f32) = b[i + 0 ..][0..8].*;
        acc0 += va0 * vb0;

        const va1: @Vector(8, f32) = a[i + 8 ..][0..8].*;
        const vb1: @Vector(8, f32) = b[i + 8 ..][0..8].*;
        acc1 += va1 * vb1;

        const va2: @Vector(8, f32) = a[i + 16 ..][0..8].*;
        const vb2: @Vector(8, f32) = b[i + 16 ..][0..8].*;
        acc2 += va2 * vb2;

        const va3: @Vector(8, f32) = a[i + 24 ..][0..8].*;
        const vb3: @Vector(8, f32) = b[i + 24 ..][0..8].*;
        acc3 += va3 * vb3;
    }

    var rem_acc: @Vector(8, f32) = @splat(0.0);
    while (i + 8 <= len) : (i += 8) {
        const va: @Vector(8, f32) = a[i..][0..8].*;
        const vb: @Vector(8, f32) = b[i..][0..8].*;
        rem_acc += va * vb;
    }

    var sum = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3) + rem_acc);

    // Scalar remainder
    while (i < len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

/// Fast Matrix-Vector Multiplication: y = W * x + (bias)
/// W is shape [M, K] in row-major order.
pub fn gemvF32(
    W: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
) void {
    const safe_M = @min(M, y.len);
    if (bias) |b| {
        for (0..safe_M) |r| {
            const row_start = r * K;
            if (row_start + K <= W.len) {
                const b_val = if (r < b.len) b[r] else 0.0;
                y[r] = dotProductF32(W[row_start .. row_start + K], x) + b_val;
            } else if (row_start < W.len) {
                const b_val = if (r < b.len) b[r] else 0.0;
                y[r] = dotProductF32(W[row_start..], x[0 .. W.len - row_start]) + b_val;
            } else {
                y[r] = if (r < b.len) b[r] else 0.0;
            }
        }
    } else {
        var row_start: usize = 0;
        for (0..safe_M) |r| {
            if (row_start + K <= W.len) {
                y[r] = dotProductF32(W[row_start .. row_start + K], x);
            } else if (row_start < W.len) {
                y[r] = dotProductF32(W[row_start..], x[0 .. W.len - row_start]);
            } else {
                y[r] = 0.0;
            }
            row_start += K;
        }
    }
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
        const c_row = C[m * N .. (m + 1) * N];
        for (0..K) |k| {
            const a_val = A[m * K + k];
            const b_row = B[k * N .. (k + 1) * N];
            for (0..N) |n| {
                c_row[n] += a_val * b_row[n];
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
    const safe_M = @min(M, y.len);
    const blocks_per_row = (K + block_size - 1) / block_size;
    fusedGemvNF4Worker(packed_W, scales, x, bias, y, 0, safe_M, K, block_size, blocks_per_row);
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
    const safe_M = @min(M, y.len);
    const blocks_per_row = (K + block_size - 1) / block_size;
    fusedGemvDQ8Worker(W_i8, scales, x, bias, y, 0, safe_M, K, block_size, blocks_per_row);
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

/// Element-wise Hadamard product: out[i] = a[i] * b[i] (SIMD accelerated)
pub fn elementWiseMulF32(a: []const f32, b: []const f32, out: []f32) void {
    const len = @min(@min(a.len, b.len), out.len);
    var i: usize = 0;
    while (i + 16 <= len) : (i += 16) {
        const va0: @Vector(8, f32) = a[i + 0 ..][0..8].*;
        const vb0: @Vector(8, f32) = b[i + 0 ..][0..8].*;
        const p0: [8]f32 = va0 * vb0;
        out[i + 0 ..][0..8].* = p0;

        const va1: @Vector(8, f32) = a[i + 8 ..][0..8].*;
        const vb1: @Vector(8, f32) = b[i + 8 ..][0..8].*;
        const p1: [8]f32 = va1 * vb1;
        out[i + 8 ..][0..8].* = p1;
    }
    while (i + 8 <= len) : (i += 8) {
        const va: @Vector(8, f32) = a[i..][0..8].*;
        const vb: @Vector(8, f32) = b[i..][0..8].*;
        const p: [8]f32 = va * vb;
        out[i..][0..8].* = p;
    }
    while (i < len) : (i += 1) {
        out[i] = a[i] * b[i];
    }
}

/// Fast RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight (SIMD accelerated)
pub fn rmsNormF32(
    x: []const f32,
    weight: []const f32,
    eps: f32,
    out: []f32,
) void {
    const len = @min(@min(x.len, weight.len), out.len);
    if (len == 0) return;

    const sum_sq = dotProductF32(x[0..len], x[0..len]);
    const mean_sq = sum_sq / @as(f32, @floatFromInt(len));
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);
    const v_scale: @Vector(8, f32) = @splat(inv_rms);

    var i: usize = 0;
    while (i + 16 <= len) : (i += 16) {
        const vx0: @Vector(8, f32) = x[i + 0 ..][0..8].*;
        const vw0: @Vector(8, f32) = weight[i + 0 ..][0..8].*;
        const p0: [8]f32 = vx0 * v_scale * vw0;
        out[i + 0 ..][0..8].* = p0;

        const vx1: @Vector(8, f32) = x[i + 8 ..][0..8].*;
        const vw1: @Vector(8, f32) = weight[i + 8 ..][0..8].*;
        const p1: [8]f32 = vx1 * v_scale * vw1;
        out[i + 8 ..][0..8].* = p1;
    }
    while (i + 8 <= len) : (i += 8) {
        const vx: @Vector(8, f32) = x[i..][0..8].*;
        const vw: @Vector(8, f32) = weight[i..][0..8].*;
        const p: [8]f32 = vx * v_scale * vw;
        out[i..][0..8].* = p;
    }
    while (i < len) : (i += 1) {
        out[i] = x[i] * inv_rms * weight[i];
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
    for (0..data.len) |i| {
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
        const qs_i8: [*]const i8 = @ptrCast(qs_bytes.ptr);
        const block_sum = dotProductInt8F32(qs_i8[0..32], x_sub);
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
    const safe_M = @min(M, y.len);
    const blocks_per_row = K / 32;
    const row_bytes_len = blocks_per_row * 34;
    for (0..safe_M) |r| {
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
    const safe_M = @min(M, y.len);
    const blocks_per_row = K / 32;
    const row_bytes_len = blocks_per_row * 18;
    for (0..safe_M) |r| {
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
    const safe_M = @min(M, y.len);
    const superblocks_per_row = K / 256;
    const row_bytes_len = superblocks_per_row * 144;
    var deq_buf: [256]f32 = undefined;
    var blk: quantization.BlockQ4_K = undefined;
    const blk_slice = std.mem.asBytes(&blk);

    for (0..safe_M) |r| {
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

/// Convert BF16 (represented as u16 raw bits) to standard IEEE 754 f32 with 0 compute overhead
pub inline fn bf16ToF32(val: u16) f32 {
    const u: u32 = @as(u32, val) << 16;
    return @bitCast(u);
}

/// Fast SIMD dot product of raw BF16 row with f32 activation vector
/// 4 independent 8-wide vector accumulators (32 elements/iter) saturate AMD Zen dual-FMA and Intel/ARM units
pub fn dotProductBF16(a: []const u16, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    var i: usize = 0;

    var acc0: @Vector(8, f32) = @splat(0.0);
    var acc1: @Vector(8, f32) = @splat(0.0);
    var acc2: @Vector(8, f32) = @splat(0.0);
    var acc3: @Vector(8, f32) = @splat(0.0);

    while (i + 32 <= len) : (i += 32) {
        const v0_u16: @Vector(8, u16) = a[i + 0 ..][0..8].*;
        const v0_f32: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), v0_u16) << @splat(16));
        const vb0: @Vector(8, f32) = b[i + 0 ..][0..8].*;
        acc0 += v0_f32 * vb0;

        const v1_u16: @Vector(8, u16) = a[i + 8 ..][0..8].*;
        const v1_f32: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), v1_u16) << @splat(16));
        const vb1: @Vector(8, f32) = b[i + 8 ..][0..8].*;
        acc1 += v1_f32 * vb1;

        const v2_u16: @Vector(8, u16) = a[i + 16 ..][0..8].*;
        const v2_f32: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), v2_u16) << @splat(16));
        const vb2: @Vector(8, f32) = b[i + 16 ..][0..8].*;
        acc2 += v2_f32 * vb2;

        const v3_u16: @Vector(8, u16) = a[i + 24 ..][0..8].*;
        const v3_f32: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), v3_u16) << @splat(16));
        const vb3: @Vector(8, f32) = b[i + 24 ..][0..8].*;
        acc3 += v3_f32 * vb3;
    }

    var sum = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));

    while (i + 8 <= len) : (i += 8) {
        const v_u16: @Vector(8, u16) = a[i..][0..8].*;
        const v_f32: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), v_u16) << @splat(16));
        const vb: @Vector(8, f32) = b[i..][0..8].*;
        sum += @reduce(.Add, v_f32 * vb);
    }

    while (i < len) : (i += 1) {
        sum += bf16ToF32(a[i]) * b[i];
    }
    return sum;
}

fn gemvBF16_4rows(
    w0: []const u16,
    w1: []const u16,
    w2: []const u16,
    w3: []const u16,
    x: []const f32,
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    out: *[4]f32,
) void {
    const len = @min(@min(@min(w0.len, w1.len), @min(w2.len, w3.len)), x.len);
    var i: usize = 0;

    var acc0_0: @Vector(8, f32) = @splat(0.0);
    var acc0_1: @Vector(8, f32) = @splat(0.0);
    var acc1_0: @Vector(8, f32) = @splat(0.0);
    var acc1_1: @Vector(8, f32) = @splat(0.0);
    var acc2_0: @Vector(8, f32) = @splat(0.0);
    var acc2_1: @Vector(8, f32) = @splat(0.0);
    var acc3_0: @Vector(8, f32) = @splat(0.0);
    var acc3_1: @Vector(8, f32) = @splat(0.0);

    while (i + 16 <= len) : (i += 16) {
        // Chunk 0 (8 elements)
        const vx0: @Vector(8, f32) = x[i..][0..8].*;
        const r0_0: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w0[i..][0..8].*) << @splat(16));
        const r1_0: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w1[i..][0..8].*) << @splat(16));
        const r2_0: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w2[i..][0..8].*) << @splat(16));
        const r3_0: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w3[i..][0..8].*) << @splat(16));

        acc0_0 += r0_0 * vx0;
        acc1_0 += r1_0 * vx0;
        acc2_0 += r2_0 * vx0;
        acc3_0 += r3_0 * vx0;

        // Chunk 1 (8 elements)
        const vx1: @Vector(8, f32) = x[i + 8 ..][0..8].*;
        const r0_1: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w0[i + 8 ..][0..8].*) << @splat(16));
        const r1_1: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w1[i + 8 ..][0..8].*) << @splat(16));
        const r2_1: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w2[i + 8 ..][0..8].*) << @splat(16));
        const r3_1: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w3[i + 8 ..][0..8].*) << @splat(16));

        acc0_1 += r0_1 * vx1;
        acc1_1 += r1_1 * vx1;
        acc2_1 += r2_1 * vx1;
        acc3_1 += r3_1 * vx1;
    }

    var sum0 = @reduce(.Add, acc0_0 + acc0_1);
    var sum1 = @reduce(.Add, acc1_0 + acc1_1);
    var sum2 = @reduce(.Add, acc2_0 + acc2_1);
    var sum3 = @reduce(.Add, acc3_0 + acc3_1);

    while (i + 8 <= len) : (i += 8) {
        const vx: @Vector(8, f32) = x[i..][0..8].*;
        const r0: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w0[i..][0..8].*) << @splat(16));
        const r1: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w1[i..][0..8].*) << @splat(16));
        const r2: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w2[i..][0..8].*) << @splat(16));
        const r3: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), w3[i..][0..8].*) << @splat(16));

        sum0 += @reduce(.Add, r0 * vx);
        sum1 += @reduce(.Add, r1 * vx);
        sum2 += @reduce(.Add, r2 * vx);
        sum3 += @reduce(.Add, r3 * vx);
    }

    while (i < len) : (i += 1) {
        const xi = x[i];
        sum0 += bf16ToF32(w0[i]) * xi;
        sum1 += bf16ToF32(w1[i]) * xi;
        sum2 += bf16ToF32(w2[i]) * xi;
        sum3 += bf16ToF32(w3[i]) * xi;
    }

    out[0] = sum0 + b0;
    out[1] = sum1 + b1;
    out[2] = sum2 + b2;
    out[3] = sum3 + b3;
}

/// Fast Matrix-Vector Multiplication for Raw BF16 Weights: y = W_bf16 * x + bias
/// Direct zero-copy computation without prior dequantization or transcoding
pub fn gemvBF16(
    W_bf16: []const u16,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
) void {
    const safe_M = @min(M, y.len);
    var r: usize = 0;
    while (r + 4 <= safe_M and (r + 4) * K <= W_bf16.len) : (r += 4) {
        const row0 = W_bf16[(r + 0) * K .. (r + 1) * K];
        const row1 = W_bf16[(r + 1) * K .. (r + 2) * K];
        const row2 = W_bf16[(r + 2) * K .. (r + 3) * K];
        const row3 = W_bf16[(r + 3) * K .. (r + 4) * K];

        const b0: f32 = if (bias) |b| (if (r + 0 < b.len) b[r + 0] else 0.0) else 0.0;
        const b1: f32 = if (bias) |b| (if (r + 1 < b.len) b[r + 1] else 0.0) else 0.0;
        const b2: f32 = if (bias) |b| (if (r + 2 < b.len) b[r + 2] else 0.0) else 0.0;
        const b3: f32 = if (bias) |b| (if (r + 3 < b.len) b[r + 3] else 0.0) else 0.0;

        var out: [4]f32 = undefined;
        gemvBF16_4rows(row0, row1, row2, row3, x, b0, b1, b2, b3, &out);
        y[r + 0] = out[0];
        y[r + 1] = out[1];
        y[r + 2] = out[2];
        y[r + 3] = out[3];
    }

    while (r < safe_M) : (r += 1) {
        const row_start = r * K;
        const row_end = @min(row_start + K, W_bf16.len);
        if (row_start >= W_bf16.len) {
            y[r] = if (bias) |b| (if (r < b.len) b[r] else 0.0) else 0.0;
            continue;
        }
        const row = W_bf16[row_start..row_end];
        var dot = dotProductBF16(row, x);
        if (bias) |b| {
            if (r < b.len) dot += b[r];
        }
        y[r] = dot;
    }
}

/// Fast SIMD dot product of raw FP16 row with f32 activation vector (32 elements/iter)
pub fn dotProductF16(a: []const f16, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    var i: usize = 0;

    var acc0: @Vector(8, f32) = @splat(0.0);
    var acc1: @Vector(8, f32) = @splat(0.0);
    var acc2: @Vector(8, f32) = @splat(0.0);
    var acc3: @Vector(8, f32) = @splat(0.0);

    while (i + 32 <= len) : (i += 32) {
        const v0_f16: @Vector(8, f16) = a[i + 0 ..][0..8].*;
        const v0_f32: @Vector(8, f32) = @floatCast(v0_f16);
        const vb0: @Vector(8, f32) = b[i + 0 ..][0..8].*;
        acc0 += v0_f32 * vb0;

        const v1_f16: @Vector(8, f16) = a[i + 8 ..][0..8].*;
        const v1_f32: @Vector(8, f32) = @floatCast(v1_f16);
        const vb1: @Vector(8, f32) = b[i + 8 ..][0..8].*;
        acc1 += v1_f32 * vb1;

        const v2_f16: @Vector(8, f16) = a[i + 16 ..][0..8].*;
        const v2_f32: @Vector(8, f32) = @floatCast(v2_f16);
        const vb2: @Vector(8, f32) = b[i + 16 ..][0..8].*;
        acc2 += v2_f32 * vb2;

        const v3_f16: @Vector(8, f16) = a[i + 24 ..][0..8].*;
        const v3_f32: @Vector(8, f32) = @floatCast(v3_f16);
        const vb3: @Vector(8, f32) = b[i + 24 ..][0..8].*;
        acc3 += v3_f32 * vb3;
    }

    var sum = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));

    while (i + 8 <= len) : (i += 8) {
        const v_f16: @Vector(8, f16) = a[i..][0..8].*;
        const v_f32: @Vector(8, f32) = @floatCast(v_f16);
        const vb: @Vector(8, f32) = b[i..][0..8].*;
        sum += @reduce(.Add, v_f32 * vb);
    }

    while (i < len) : (i += 1) {
        sum += @as(f32, @floatCast(a[i])) * b[i];
    }
    return sum;
}

/// Fast Matrix-Vector Multiplication for Raw FP16 Weights: y = W_f16 * x + bias
pub fn gemvF16(
    W_f16: []const f16,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
) void {
    const safe_M = @min(M, y.len);
    for (0..safe_M) |r| {
        const row_start = r * K;
        const row_end = @min(row_start + K, W_f16.len);
        if (row_start >= W_f16.len) {
            y[r] = if (bias) |b| (if (r < b.len) b[r] else 0.0) else 0.0;
            continue;
        }
        const row = W_f16[row_start..row_end];
        var dot = dotProductF16(row, x);
        if (bias) |b| {
            if (r < b.len) dot += b[r];
        }
        y[r] = dot;
    }
}

/// Fast INT8 dot product (optimized for Intel VNNI and ARM NEON pipelines, 32 elements/iter)
pub fn dotProductInt8(a: []const i8, b: []const i8) i32 {
    const len = @min(a.len, b.len);
    var i: usize = 0;

    var acc0: @Vector(8, i32) = @splat(0);
    var acc1: @Vector(8, i32) = @splat(0);
    var acc2: @Vector(8, i32) = @splat(0);
    var acc3: @Vector(8, i32) = @splat(0);

    while (i + 32 <= len) : (i += 32) {
        const v0_a: @Vector(8, i8) = a[i + 0 ..][0..8].*;
        const v0_b: @Vector(8, i8) = b[i + 0 ..][0..8].*;
        acc0 += @as(@Vector(8, i32), v0_a) * @as(@Vector(8, i32), v0_b);

        const v1_a: @Vector(8, i8) = a[i + 8 ..][0..8].*;
        const v1_b: @Vector(8, i8) = b[i + 8 ..][0..8].*;
        acc1 += @as(@Vector(8, i32), v1_a) * @as(@Vector(8, i32), v1_b);

        const v2_a: @Vector(8, i8) = a[i + 16 ..][0..8].*;
        const v2_b: @Vector(8, i8) = b[i + 16 ..][0..8].*;
        acc2 += @as(@Vector(8, i32), v2_a) * @as(@Vector(8, i32), v2_b);

        const v3_a: @Vector(8, i8) = a[i + 24 ..][0..8].*;
        const v3_b: @Vector(8, i8) = b[i + 24 ..][0..8].*;
        acc3 += @as(@Vector(8, i32), v3_a) * @as(@Vector(8, i32), v3_b);
    }

    var sum = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));

    while (i + 8 <= len) : (i += 8) {
        const va: @Vector(8, i8) = a[i..][0..8].*;
        const vb: @Vector(8, i8) = b[i..][0..8].*;
        sum += @reduce(.Add, @as(@Vector(8, i32), va) * @as(@Vector(8, i32), vb));
    }

    while (i < len) : (i += 1) {
        sum += @as(i32, a[i]) * @as(i32, b[i]);
    }
    return sum;
}

/// Fast SIMD dot product of INT8 weights with float32 activations (32 elements/iter)
pub fn dotProductInt8F32(a: []const i8, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    var i: usize = 0;

    var acc0: @Vector(8, f32) = @splat(0.0);
    var acc1: @Vector(8, f32) = @splat(0.0);
    var acc2: @Vector(8, f32) = @splat(0.0);
    var acc3: @Vector(8, f32) = @splat(0.0);

    while (i + 32 <= len) : (i += 32) {
        const v0_i8: @Vector(8, i8) = a[i + 0 ..][0..8].*;
        const vb0: @Vector(8, f32) = b[i + 0 ..][0..8].*;
        acc0 += @as(@Vector(8, f32), @floatFromInt(v0_i8)) * vb0;

        const v1_i8: @Vector(8, i8) = a[i + 8 ..][0..8].*;
        const vb1: @Vector(8, f32) = b[i + 8 ..][0..8].*;
        acc1 += @as(@Vector(8, f32), @floatFromInt(v1_i8)) * vb1;

        const v2_i8: @Vector(8, i8) = a[i + 16 ..][0..8].*;
        const vb2: @Vector(8, f32) = b[i + 16 ..][0..8].*;
        acc2 += @as(@Vector(8, f32), @floatFromInt(v2_i8)) * vb2;

        const v3_i8: @Vector(8, i8) = a[i + 24 ..][0..8].*;
        const vb3: @Vector(8, f32) = b[i + 24 ..][0..8].*;
        acc3 += @as(@Vector(8, f32), @floatFromInt(v3_i8)) * vb3;
    }

    var sum = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));

    while (i + 8 <= len) : (i += 8) {
        const v_i8: @Vector(8, i8) = a[i..][0..8].*;
        const vb: @Vector(8, f32) = b[i..][0..8].*;
        sum += @reduce(.Add, @as(@Vector(8, f32), @floatFromInt(v_i8)) * vb);
    }

    while (i < len) : (i += 1) {
        sum += @as(f32, @floatFromInt(a[i])) * b[i];
    }
    return sum;
}

/// Fast Matrix-Vector Multiplication for Raw INT8 Weights with f32 activation (SIMD accelerated):
/// y = (W_int8 * x) * scale + bias
pub fn gemvInt8Scaled(
    W_i8: []const i8,
    x: []const f32,
    scale_w: f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
) void {
    const safe_M = @min(M, y.len);
    for (0..safe_M) |r| {
        const row_start = r * K;
        const row_end = @min(row_start + K, W_i8.len);
        if (row_start >= W_i8.len) {
            y[r] = if (bias) |b| (if (r < b.len) b[r] else 0.0) else 0.0;
            continue;
        }
        const row = W_i8[row_start..row_end];
        var row_sum = dotProductInt8F32(row, x) * scale_w;
        if (bias) |b| {
            if (r < b.len) row_sum += b[r];
        }
        y[r] = row_sum;
    }
}


