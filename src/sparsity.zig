const std = @import("std");

/// Decodes bitmask-compressed tensor elements to dense FP32.
/// Input payload layout:
///   [bitmask: ceil(count / 8) bytes] [padding to 4 bytes] [non_zero_values: nnz * 4 bytes (f32)]
pub fn decodeBitmaskF32(
    payload: []const u8,
    count: usize,
    out: []f32,
) !void {
    const mask_bytes_len = (count + 7) / 8;
    const val_offset = (mask_bytes_len + 3) & ~@as(usize, 3);
    if (payload.len < val_offset) return error.PayloadTooShort;

    const mask = payload[0..mask_bytes_len];
    const val_bytes = payload[val_offset..];
    const num_floats = val_bytes.len / 4;

    var val_idx: usize = 0;
    for (0..count) |i| {
        const byte_idx = i / 8;
        const bit_idx: u3 = @intCast(i % 8);
        const is_non_zero = ((mask[byte_idx] >> bit_idx) & 1) != 0;

        if (is_non_zero) {
            if (val_idx >= num_floats) return error.BitmaskValueCountMismatch;
            const u32_bits = std.mem.readInt(u32, val_bytes[val_idx * 4 ..][0..4], .little);
            out[i] = @bitCast(u32_bits);
            val_idx += 1;
        } else {
            out[i] = 0.0;
        }
    }
}

/// Encodes dense FP32 elements into bitmask-compressed format.
pub fn encodeBitmaskF32(
    dense_in: []const f32,
    allocator: std.mem.Allocator,
) ![]u8 {
    const count = dense_in.len;
    const mask_bytes_len = (count + 7) / 8;
    const val_offset = (mask_bytes_len + 3) & ~@as(usize, 3);

    var nnz: usize = 0;
    for (dense_in) |x| {
        if (x != 0.0) nnz += 1;
    }

    const total_bytes = val_offset + (nnz * @sizeOf(f32));
    const out_buf = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(out_buf);

    @memset(out_buf[0..val_offset], 0);

    var val_idx: usize = 0;
    for (0..count) |i| {
        if (dense_in[i] != 0.0) {
            const byte_idx = i / 8;
            const bit_idx: u3 = @intCast(i % 8);
            out_buf[byte_idx] |= (@as(u8, 1) << bit_idx);

            const bits: u32 = @bitCast(dense_in[i]);
            std.mem.writeInt(u32, out_buf[val_offset + val_idx * 4 ..][0..4], bits, .little);
            val_idx += 1;
        }
    }

    return out_buf;
}

/// Unpacks NVIDIA Ampere 2:4 structured sparse payload into dense FP32.
/// Input payload format:
///   [meta: ceil(count / 8) bytes] [padding to 4-byte boundary] [values: (count / 2) * 4 bytes (f32)]
pub fn decodeStructured2_4_F32(
    payload: []const u8,
    count: usize,
    out: []f32,
) !void {
    if (count % 4 != 0) return error.InvalidStructuredCount;

    const num_groups = count / 4;
    const meta_len = (num_groups + 1) / 2;
    const val_offset = (meta_len + 3) & ~@as(usize, 3);
    const expected_val_bytes = (count / 2) * @sizeOf(f32);

    if (payload.len < val_offset + expected_val_bytes) return error.PayloadTooShort;

    const meta = payload[0..meta_len];
    const val_bytes = payload[val_offset .. val_offset + expected_val_bytes];

    @memset(out, 0.0);
    const is_aligned = std.mem.isAligned(@intFromPtr(val_bytes.ptr), @alignOf(f32));
    const val_ptr: ?[*]const f32 = if (is_aligned) @ptrCast(@alignCast(val_bytes.ptr)) else null;
    const full_pairs = num_groups / 2;

    const readVal = struct {
        inline fn get(p: ?[*]const f32, bytes: []const u8, idx: usize) f32 {
            if (p) |ptr| return ptr[idx];
            const u = std.mem.readInt(u32, bytes[idx * 4 .. (idx + 1) * 4][0..4], .little);
            return @bitCast(u);
        }
    }.get;

    for (0..full_pairs) |byte_idx| {
        const meta_byte = meta[byte_idx];
        const nibble0: u4 = @truncate(meta_byte & 0x0F);
        const nibble1: u4 = @truncate((meta_byte >> 4) & 0x0F);

        const g0 = byte_idx * 2;
        const g1 = g0 + 1;

        const base0 = g0 * 4;
        const base1 = g1 * 4;

        const idx0_0: usize = (nibble0 & 0x03);
        const idx0_1: usize = ((nibble0 >> 2) & 0x03);
        out[base0 + idx0_0] = readVal(val_ptr, val_bytes, g0 * 2 + 0);
        out[base0 + idx0_1] = readVal(val_ptr, val_bytes, g0 * 2 + 1);

        const idx1_0: usize = (nibble1 & 0x03);
        const idx1_1: usize = ((nibble1 >> 2) & 0x03);
        out[base1 + idx1_0] = readVal(val_ptr, val_bytes, g1 * 2 + 0);
        out[base1 + idx1_1] = readVal(val_ptr, val_bytes, g1 * 2 + 1);
    }

    if (num_groups % 2 != 0) {
        const g = num_groups - 1;
        const meta_byte = meta[g / 2];
        const nibble0: u4 = @truncate(meta_byte & 0x0F);
        const idx0: usize = (nibble0 & 0x03);
        const idx1: usize = ((nibble0 >> 2) & 0x03);
        const base = g * 4;
        out[base + idx0] = readVal(val_ptr, val_bytes, g * 2 + 0);
        out[base + idx1] = readVal(val_ptr, val_bytes, g * 2 + 1);
    }
}

/// Packs dense 2:4 structured sparse FP32 elements into Ampere 2:4 hardware payload.
pub fn encodeStructured2_4_F32(
    dense_in: []const f32,
    allocator: std.mem.Allocator,
) ![]u8 {
    if (dense_in.len % 4 != 0) return error.InvalidStructuredCount;

    const count = dense_in.len;
    const num_groups = count / 4;
    const meta_len = (num_groups + 1) / 2;
    const val_offset = (meta_len + 3) & ~@as(usize, 3);
    const val_bytes_len = (count / 2) * @sizeOf(f32);
    const total_len = val_offset + val_bytes_len;

    const out_buf = try allocator.alloc(u8, total_len);
    errdefer allocator.free(out_buf);

    @memset(out_buf[0..val_offset], 0);

    for (0..num_groups) |g| {
        const base = g * 4;
        const block = dense_in[base..][0..4];

        var idx0: u2 = 0;
        var idx1: u2 = 1;
        var found: usize = 0;

        for (0..4) |pos| {
            if (block[pos] != 0.0) {
                if (found == 0) {
                    idx0 = @intCast(pos);
                    found += 1;
                } else if (found == 1) {
                    idx1 = @intCast(pos);
                    found += 1;
                    break;
                }
            }
        }

        if (found == 1) {
            idx1 = if (idx0 == 0) 1 else 0;
        }

        const bits0: u32 = @bitCast(block[idx0]);
        const bits1: u32 = @bitCast(block[idx1]);
        std.mem.writeInt(u32, out_buf[val_offset + (g * 2 + 0) * 4 ..][0..4], bits0, .little);
        std.mem.writeInt(u32, out_buf[val_offset + (g * 2 + 1) * 4 ..][0..4], bits1, .little);

        const nibble: u4 = (@as(u4, idx0) & 0x03) | ((@as(u4, idx1) & 0x03) << 2);
        const meta_idx = g / 2;
        if (g % 2 == 0) {
            out_buf[meta_idx] |= @as(u8, nibble);
        } else {
            out_buf[meta_idx] |= (@as(u8, nibble) << 4);
        }
    }

    return out_buf;
}

/// Unpacks Block Sparse Row (BSR) payload into dense matrix [M, K].
pub fn decodeBSR_F32(
    payload: []const u8,
    M: usize,
    K: usize,
    out: []f32,
) !void {
    if (payload.len < 12) return error.PayloadTooShort;

    const block_h = std.mem.readInt(u16, payload[0..2], .little);
    const block_w = std.mem.readInt(u16, payload[2..4], .little);
    const num_row_ptrs = std.mem.readInt(u32, payload[4..8], .little);
    const num_col_indices = std.mem.readInt(u32, payload[8..12], .little);

    const bh: usize = block_h;
    const bw: usize = block_w;
    if (bh == 0 or bw == 0) return error.InvalidBlockDimensions;

    var pos: usize = 12;
    const row_ptrs_bytes = num_row_ptrs * @sizeOf(u32);
    if (pos + row_ptrs_bytes > payload.len) return error.PayloadTooShort;
    const row_ptrs_raw = payload[pos .. pos + row_ptrs_bytes];
    pos += row_ptrs_bytes;

    const col_indices_bytes = num_col_indices * @sizeOf(u32);
    if (pos + col_indices_bytes > payload.len) return error.PayloadTooShort;
    const col_indices_raw = payload[pos .. pos + col_indices_bytes];
    pos += col_indices_bytes;

    const num_blocks: usize = num_col_indices;
    const values_bytes = num_blocks * bh * bw * @sizeOf(f32);
    if (pos + values_bytes > payload.len) return error.PayloadTooShort;
    const values_raw = payload[pos .. pos + values_bytes];

    @memset(out, 0.0);

    const num_block_rows = M / bh;
    for (0..num_block_rows) |br| {
        if (br + 1 >= num_row_ptrs) break;
        const start_b = std.mem.readInt(u32, row_ptrs_raw[br * 4 ..][0..4], .little);
        const end_b = std.mem.readInt(u32, row_ptrs_raw[(br + 1) * 4 ..][0..4], .little);

        var b = start_b;
        while (b < end_b) : (b += 1) {
            if (b >= num_col_indices) break;
            const bc: usize = std.mem.readInt(u32, col_indices_raw[b * 4 ..][0..4], .little);
            const block_start_byte = b * bh * bw * 4;

            for (0..bh) |r| {
                const global_r = br * bh + r;
                if (global_r >= M) continue;
                for (0..bw) |c| {
                    const global_c = bc * bw + c;
                    if (global_c >= K) continue;
                    const elem_byte = block_start_byte + (r * bw + c) * 4;
                    const bits = std.mem.readInt(u32, values_raw[elem_byte..][0..4], .little);
                    out[global_r * K + global_c] = @bitCast(bits);
                }
            }
        }
    }
}
