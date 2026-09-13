const std = @import("std");

/// 16-point NormalFloat-4 (NF4) quantile lookup table
/// Normalized to [-1.0, 1.0] from standard normal distribution N(0, 1) quantiles
pub const NF4_TABLE: [16]f32 = .{
    -1.0000000,
    -0.6961928,
    -0.5250731,
    -0.3949175,
    -0.2844414,
    -0.1847734,
    -0.0910500,
    0.0000000,
    0.0795803,
    0.1609302,
    0.2461123,
    0.3379152,
    0.4407098,
    0.5626170,
    0.7229568,
    1.0000000,
};

/// Finds the closest NF4 code (0..15) for a normalized value in [-1.0, 1.0]
pub fn findClosestNF4(val: f32) u4 {
    var best_idx: u4 = 0;
    var min_dist: f32 = std.math.inf(f32);

    inline for (0..16) |i| {
        const dist = @abs(val - NF4_TABLE[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = @intCast(i);
        }
    }
    return best_idx;
}

/// Quantizes a block of floats (e.g. 32 elements) to NF4 with dual-mode residual tracking.
/// packed_out receives ceil(block.len / 2) bytes.
/// residual_out (if non-null) receives exact difference (orig - dequantized) for precision recovery.
pub fn quantizeBlockNF4(
    block: []const f32,
    packed_out: []u8,
    residual_out: ?[]f32,
) f32 {
    if (block.len == 0) return 0.0;

    // Find max absolute value for scale
    var max_abs: f32 = 0.0;
    for (block) |x| {
        const a = @abs(x);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) max_abs = 1e-8;

    const inv_scale = 1.0 / max_abs;

    for (0..block.len) |i| {
        const norm = std.math.clamp(block[i] * inv_scale, -1.0, 1.0);
        const code = findClosestNF4(norm);

        const byte_idx = i / 2;
        if (i % 2 == 0) {
            packed_out[byte_idx] = @as(u8, code);
        } else {
            packed_out[byte_idx] |= (@as(u8, code) << 4);
        }

        // Track residual delta for precision recovery
        if (residual_out) |res| {
            const deq_val = NF4_TABLE[code] * max_abs;
            res[i] = block[i] - deq_val;
        }
    }

    return max_abs;
}

/// Dequantizes packed 4-bit NF4 bytes back to 32-bit floats using block scale.
pub fn dequantizeBlockNF4(
    packed_in: []const u8,
    scale: f32,
    count: usize,
    out: []f32,
) void {
    for (0..count) |i| {
        const byte_idx = i / 2;
        const code: u4 = if (i % 2 == 0)
            @truncate(packed_in[byte_idx] & 0x0F)
        else
            @truncate((packed_in[byte_idx] >> 4) & 0x0F);

        out[i] = NF4_TABLE[code] * scale;
    }
}

/// Dual-mode precision recovery: reconstructs full-fidelity weights by combining
/// compact NF4 base with residual buffer.
pub fn dequantizeWithResidual(
    packed_in: []const u8,
    scale: f32,
    residuals: []const f32,
    count: usize,
    out: []f32,
) void {
    dequantizeBlockNF4(packed_in, scale, count, out);
    for (0..count) |i| {
        out[i] += residuals[i];
    }
}
