const std = @import("std");
const format = @import("format.zig");
const reader_mod = @import("reader.zig");
const writer_mod = @import("writer.zig");
const appendix_mod = @import("appendix.zig");
const nf4 = @import("nf4.zig");
const quantization = @import("quantization.zig");
const sparsity = @import("sparsity.zig");
const tiling = @import("tiling.zig");
const tensor_ops_mod = @import("tensor_ops.zig");
const growth_mod = @import("growth.zig");

pub const hk_reader_t = opaque {};

const ReaderWrapper = struct {
    reader: reader_mod.HKReader,
    appendix_reader: ?appendix_mod.AppendixReader = null,
    allocator: std.mem.Allocator,
};

pub export fn hk_open(path_c: [*:0]const u8) ?*hk_reader_t {
    const path = std.mem.sliceTo(path_c, 0);
    const allocator = std.heap.page_allocator;

    const wrapper = allocator.create(ReaderWrapper) catch return null;
    wrapper.allocator = allocator;
    wrapper.reader = reader_mod.HKReader.open(path, allocator) catch {
        allocator.destroy(wrapper);
        return null;
    };
    wrapper.appendix_reader = appendix_mod.AppendixReader.init(allocator, wrapper.reader.mmap_region.bytes) catch null;

    return @ptrCast(wrapper);
}

pub export fn hk_close(reader_ptr: ?*hk_reader_t) void {
    if (reader_ptr == null) return;
    const wrapper: *ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (wrapper.appendix_reader) |*ar| {
        ar.deinit();
    }
    wrapper.reader.deinit();
    wrapper.allocator.destroy(wrapper);
}

pub export fn hk_get_tensor_count(reader_ptr: ?*const hk_reader_t) u64 {
    if (reader_ptr == null) return 0;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    return wrapper.reader.toc.entries.items.len;
}

pub const C_TensorInfo = extern struct {
    name: [*:0]const u8,
    storage_type: u8,
    tile_layout: u8,
    sparsity_type: u8,
    ndim: u8,
    shape: [8]u64,
    data_offset: u64,
    data_size: u64,
    residual_offset: u64,
    residual_size: u64,
    scale_offset: u64,
    scale_size: u64,
    block_size: u16,
    sparsity_ratio: f32,
};

pub export fn hk_get_tensor_info(reader_ptr: ?*const hk_reader_t, index: u64, out_info: ?*C_TensorInfo) c_int {
    if (reader_ptr == null or out_info == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (index >= wrapper.reader.toc.entries.items.len) return -1;

    const entry = wrapper.reader.toc.entries.items[index];
    out_info.?.* = .{
        .name = @ptrCast(entry.name.ptr),
        .storage_type = @intFromEnum(entry.storage_type),
        .tile_layout = @intFromEnum(entry.tile_layout),
        .sparsity_type = @intFromEnum(entry.sparsity_type),
        .ndim = entry.ndim,
        .shape = entry.shape,
        .data_offset = entry.data_offset,
        .data_size = entry.data_size,
        .residual_offset = entry.residual_offset,
        .residual_size = entry.residual_size,
        .scale_offset = entry.scale_offset,
        .scale_size = entry.scale_size,
        .block_size = entry.block_size,
        .sparsity_ratio = entry.sparsity_ratio,
    };
    return 0;
}

pub export fn hk_get_tensor_data(reader_ptr: ?*const hk_reader_t, index: u64, out_size: ?*u64) ?*const anyopaque {
    if (reader_ptr == null) return null;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (index >= wrapper.reader.toc.entries.items.len) return null;

    const entry = wrapper.reader.toc.entries.items[index];
    const data = wrapper.reader.getTensorData(entry) catch return null;
    if (out_size) |sz| {
        sz.* = data.len;
    }
    return @ptrCast(data.ptr);
}

pub export fn hk_get_tensor_residual(reader_ptr: ?*const hk_reader_t, index: u64, out_size: ?*u64) ?*const anyopaque {
    if (reader_ptr == null) return null;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (index >= wrapper.reader.toc.entries.items.len) return null;

    const entry = wrapper.reader.toc.entries.items[index];
    const res = wrapper.reader.getTensorResidual(entry) orelse return null;
    if (out_size) |sz| {
        sz.* = res.len;
    }
    return @ptrCast(res.ptr);
}

pub export fn hk_get_tensor_scales(reader_ptr: ?*const hk_reader_t, index: u64, out_size: ?*u64) ?*const anyopaque {
    if (reader_ptr == null) return null;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (index >= wrapper.reader.toc.entries.items.len) return null;

    const entry = wrapper.reader.toc.entries.items[index];
    const sc = wrapper.reader.getTensorScales(entry) orelse return null;
    if (out_size) |sz| {
        sz.* = sc.len;
    }
    return @ptrCast(sc.ptr);
}

pub export fn hk_dequantize_f32(
    reader_ptr: ?*const hk_reader_t,
    index: u64,
    with_residual: c_int,
    out_buf: ?[*]f32,
    count: u64,
) c_int {
    if (reader_ptr == null or out_buf == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (index >= wrapper.reader.toc.entries.items.len) return -1;

    const entry = wrapper.reader.toc.entries.items[index];
    const out_slice = out_buf.?[0..count];
    wrapper.reader.dequantizeToF32(entry, with_residual != 0, out_slice) catch return -1;
    return 0;
}

pub export fn hk_get_metadata_string(reader_ptr: ?*const hk_reader_t, key_c: [*:0]const u8) ?[*:0]const u8 {
    if (reader_ptr == null) return null;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    const key = std.mem.sliceTo(key_c, 0);

    const val = wrapper.reader.metadata_map.get(key) orelse return null;
    return switch (val) {
        .val_string => |s| @ptrCast(s.ptr),
        .val_json => |j| @ptrCast(j.ptr),
        else => null,
    };
}

pub export fn hk_get_metadata_int(reader_ptr: ?*const hk_reader_t, key_c: [*:0]const u8, out_val: ?*i64) c_int {
    if (reader_ptr == null or out_val == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    const key = std.mem.sliceTo(key_c, 0);

    const val = wrapper.reader.metadata_map.get(key) orelse return -1;
    switch (val) {
        .val_int64 => |v| {
            out_val.?.* = v;
            return 0;
        },
        else => return -1,
    }
}

pub export fn hk_get_metadata_float(reader_ptr: ?*const hk_reader_t, key_c: [*:0]const u8, out_val: ?*f64) c_int {
    if (reader_ptr == null or out_val == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    const key = std.mem.sliceTo(key_c, 0);

    const val = wrapper.reader.metadata_map.get(key) orelse return -1;
    switch (val) {
        .val_float64 => |f| {
            out_val.?.* = f;
            return 0;
        },
        else => return -1,
    }
}

pub export fn hk_get_metadata_bool(reader_ptr: ?*const hk_reader_t, key_c: [*:0]const u8, out_val: ?*c_int) c_int {
    if (reader_ptr == null or out_val == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    const key = std.mem.sliceTo(key_c, 0);

    const val = wrapper.reader.metadata_map.get(key) orelse return -1;
    switch (val) {
        .val_bool => |b| {
            out_val.?.* = if (b) 1 else 0;
            return 0;
        },
        else => return -1,
    }
}

pub export fn hk_quantize_block_nf4(
    block: [*]const f32,
    count: u32,
    packed_out: [*]u8,
    residual_out: ?[*]f32,
) f32 {
    const block_slice = block[0..count];
    const packed_len = (count + 1) / 2;
    const packed_slice = packed_out[0..packed_len];
    const res_slice: ?[]f32 = if (residual_out) |r| r[0..count] else null;
    return quantization.quantizeBlockNF4(block_slice, packed_slice, res_slice);
}

pub export fn hk_quantize_block_dq8(
    block: [*]const f32,
    count: u32,
    out_i8: [*]i8,
    residual_out: ?[*]f32,
) f32 {
    const block_slice = block[0..count];
    const i8_slice = out_i8[0..count];
    const res_slice: ?[]f32 = if (residual_out) |r| r[0..count] else null;
    return quantization.quantizeBlockDQ8(block_slice, i8_slice, res_slice);
}

pub export fn hk_quantize_block_dqt(
    block: [*]const f32,
    count: u32,
    packed_out: [*]u8,
    residual_out: ?[*]f32,
) f32 {
    const block_slice = block[0..count];
    const packed_len = (count + 3) / 4;
    const packed_slice = packed_out[0..packed_len];
    const res_slice: ?[]f32 = if (residual_out) |r| r[0..count] else null;
    return quantization.quantizeBlockDQT(block_slice, packed_slice, res_slice);
}

pub export fn hk_quantize_block_q4_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ4_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ4_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_quantize_block_q4_0(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ4_0) c_int {
    if (count != quantization.QK4_0) return -1;
    quantization.quantizeBlockQ4_0(weights[0..quantization.QK4_0], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q4_0(block: *const quantization.BlockQ4_0, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeBlockQ4_0(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_quantize_block_q8_0(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ8_0) c_int {
    if (count != quantization.QK8_0) return -1;
    quantization.quantizeBlockQ8_0(weights[0..quantization.QK8_0], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q8_0(block: *const quantization.BlockQ8_0, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeBlockQ8_0(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_quantize_block_q5_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ5_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ5_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q5_k(block: *const quantization.BlockQ5_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ5_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_quantize_block_q3_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ3_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ3_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q3_k(block: *const quantization.BlockQ3_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ3_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_quantize_block_q6_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ6_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ6_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_quantize_block_q2_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ2_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ2_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q4_k(block: *const quantization.BlockQ4_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ4_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_quantize_block_q8_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ8_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ8_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q8_k(block: *const quantization.BlockQ8_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ8_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_dequantize_block_q6_k(block: *const quantization.BlockQ6_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ6_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_dequantize_block_q2_k(block: *const quantization.BlockQ2_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ2_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_dequantize_block_iq4_nl(packed_in: [*]const u8, scale: f32, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeBlockIQ4_NL(packed_in[0..(count + 1) / 2], scale, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_dequantize_block_mxfp4(packed_in: [*]const u8, scale_e8m0: u8, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeBlockMXFP4(packed_in[0..(count + 1) / 2], scale_e8m0, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_dequantize_block_nvfp4(packed_in: [*]const u8, scale_fp8: u8, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeBlockNVFP4(packed_in[0..(count + 1) / 2], scale_fp8, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_pack_2_4(
    dense_in: [*]const f32,
    count: u64,
    out_bytes: [*]u8,
) c_int {
    if (count % 4 != 0) return -1;
    const num_groups: usize = @intCast(count / 4);
    const meta_len = (num_groups + 1) / 2;
    const val_offset = (meta_len + 3) & ~@as(usize, 3);

    @memset(out_bytes[0..val_offset], 0);
    const val_out: [*]f32 = @ptrCast(@alignCast(out_bytes + val_offset));

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

        val_out[g * 2 + 0] = block[idx0];
        val_out[g * 2 + 1] = block[idx1];

        const nibble: u4 = (@as(u4, idx0) & 0x03) | ((@as(u4, idx1) & 0x03) << 2);
        const meta_idx = g / 2;
        if (g % 2 == 0) {
            out_bytes[meta_idx] |= @as(u8, nibble);
        } else {
            out_bytes[meta_idx] |= (@as(u8, nibble) << 4);
        }
    }
    return 0;
}

pub export fn hk_unpack_2_4(
    payload: [*]const u8,
    payload_len: u64,
    count: u64,
    out_buf: [*]f32,
) c_int {
    const payload_slice = payload[0..payload_len];
    const out_slice = out_buf[0..count];
    sparsity.decodeStructured2_4_F32(payload_slice, count, out_slice) catch return -1;
    return 0;
}

pub export fn hk_tile_16x16_pack(
    in_row_major: [*]const f32,
    M: u64,
    K: u64,
    out_tiled: [*]f32,
) c_int {
    const pad_m = (16 - (M % 16)) % 16;
    const pad_k = (16 - (K % 16)) % 16;
    const padded_m = M + pad_m;
    const padded_k = K + pad_k;

    const num_tiles_m = padded_m / 16;
    const num_tiles_k = padded_k / 16;
    const total_elements = num_tiles_m * num_tiles_k * 16 * 16;
    @memset(out_tiled[0..total_elements], 0.0);

    var tile_idx: usize = 0;
    for (0..num_tiles_m) |tm| {
        for (0..num_tiles_k) |tk| {
            const tile_start = tile_idx * 256;
            for (0..16) |r| {
                const global_r = tm * 16 + r;
                for (0..16) |c| {
                    const global_c = tk * 16 + c;
                    const val = if (global_r < M and global_c < K)
                        in_row_major[global_r * K + global_c]
                    else
                        0.0;
                    out_tiled[tile_start + (r * 16 + c)] = val;
                }
            }
            tile_idx += 1;
        }
    }
    return 0;
}

pub export fn hk_tile_16x16_unpack(
    in_tiled: [*]const f32,
    M: u64,
    K: u64,
    out_row_major: [*]f32,
) c_int {
    const pad_m = (16 - (M % 16)) % 16;
    const pad_k = (16 - (K % 16)) % 16;
    const padded_m = M + pad_m;
    const padded_k = K + pad_k;

    const num_tiles_m = padded_m / 16;
    const num_tiles_k = padded_k / 16;

    var tile_idx: usize = 0;
    for (0..num_tiles_m) |tm| {
        for (0..num_tiles_k) |tk| {
            const tile_start = tile_idx * 256;
            for (0..16) |r| {
                const global_r = tm * 16 + r;
                for (0..16) |c| {
                    const global_c = tk * 16 + c;
                    if (global_r < M and global_c < K) {
                        out_row_major[global_r * K + global_c] = in_tiled[tile_start + (r * 16 + c)];
                    }
                }
            }
            tile_idx += 1;
        }
    }
    return 0;
}

pub const C_AppendixEntry = extern struct {
    entry_type: u8,
    flags: u8,
    generation: u32,
    timestamp: u64,
    parent_hash: [32]u8,
    metric_loss: f32,
    metric_acc: f32,
    metric_pass: f32,
    metric_custom: f32,
    name: [*:0]const u8,
    target: [*:0]const u8,
    data: ?*const anyopaque,
    data_size: u64,
};

pub export fn hk_appendix_get_count(reader_ptr: ?*const hk_reader_t) u64 {
    if (reader_ptr == null) return 0;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (wrapper.appendix_reader) |ar| {
        return ar.records.items.len;
    }
    return 0;
}

pub export fn hk_appendix_get_entry(reader_ptr: ?*const hk_reader_t, index: u64, out_entry: ?*C_AppendixEntry) c_int {
    if (reader_ptr == null or out_entry == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (wrapper.appendix_reader == null) return -1;
    const records = wrapper.appendix_reader.?.records.items;
    if (index >= records.len) return -1;
    const rec = records[index];

    out_entry.?.* = .{
        .entry_type = @intFromEnum(rec.entry_type),
        .flags = rec.flags,
        .generation = rec.generation,
        .timestamp = rec.timestamp,
        .parent_hash = rec.parent_hash,
        .metric_loss = rec.metrics.loss,
        .metric_acc = rec.metrics.accuracy,
        .metric_pass = rec.metrics.pass_rate,
        .metric_custom = rec.metrics.custom,
        .name = @ptrCast(rec.name.ptr),
        .target = if (rec.target.len > 0) @ptrCast(rec.target.ptr) else "",
        .data = if (rec.data.len > 0) @ptrCast(rec.data.ptr) else null,
        .data_size = rec.data.len,
    };
    return 0;
}

pub export fn hk_appendix_append(
    file_path_c: [*:0]const u8,
    entry_type: u8,
    flags: u8,
    name_c: [*:0]const u8,
    target_c: [*:0]const u8,
    generation: u32,
    parent_hash_ptr: ?[*]const u8,
    metric_loss: f32,
    metric_acc: f32,
    metric_pass: f32,
    metric_custom: f32,
    data_ptr: ?[*]const u8,
    data_size: u64,
) c_int {
    const file_path = std.mem.sliceTo(file_path_c, 0);
    const name = std.mem.sliceTo(name_c, 0);
    const target = std.mem.sliceTo(target_c, 0);
    const allocator = std.heap.page_allocator;

    var ph: [32]u8 = [_]u8{0} ** 32;
    if (parent_hash_ptr != null) {
        @memcpy(&ph, parent_hash_ptr.?[0..32]);
    }

    const payload: []const u8 = if (data_ptr != null and data_size > 0)
        data_ptr.?[0..@intCast(data_size)]
    else
        &[_]u8{};

    const io = std.Options.debug_io;
    const cur_time_ns = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const cur_time_sec: u64 = @intCast(@max(0, @divTrunc(cur_time_ns, 1_000_000_000)));

    const record = format.AppendixRecord{
        .entry_type = @enumFromInt(entry_type),
        .flags = flags,
        .name = name,
        .target = target,
        .generation = generation,
        .timestamp = cur_time_sec,
        .parent_hash = ph,
        .metrics = .{
            .loss = metric_loss,
            .accuracy = metric_acc,
            .pass_rate = metric_pass,
            .custom = metric_custom,
        },
        .data = payload,
    };

    appendix_mod.appendRecordToFile(allocator, file_path, record) catch return -1;
    return 0;
}

pub export fn hk_appendix_rollback(file_path_c: [*:0]const u8, target_generation: u32) c_int {
    const file_path = std.mem.sliceTo(file_path_c, 0);
    const allocator = std.heap.page_allocator;
    appendix_mod.rollbackToFile(allocator, file_path, target_generation) catch return -1;
    return 0;
}

pub export fn hk_dot_product_f32(a: [*]const f32, b: [*]const f32, count: u64) f32 {
    return tensor_ops_mod.dotProductF32(a[0..@intCast(count)], b[0..@intCast(count)]);
}

pub export fn hk_gemv_f32(
    W: [*]const f32,
    x: [*]const f32,
    bias: ?[*]const f32,
    y: [*]f32,
    M: u64,
    K: u64,
) void {
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..@intCast(M)] else null;
    tensor_ops_mod.gemvF32(W[0..@intCast(M * K)], x[0..@intCast(K)], bias_slice, y[0..@intCast(M)], @intCast(M), @intCast(K));
}

pub export fn hk_gemm_f32(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: u64,
    K: u64,
    N: u64,
) void {
    tensor_ops_mod.gemmF32(A[0..@intCast(M * K)], B[0..@intCast(K * N)], C[0..@intCast(M * N)], @intCast(M), @intCast(K), @intCast(N));
}

pub export fn hk_fused_gemv_nf4(
    packed_W: [*]const u8,
    scales: [*]const f32,
    x: [*]const f32,
    bias: ?[*]const f32,
    y: [*]f32,
    M: u64,
    K: u64,
    block_size: u32,
) void {
    const k_bytes = (K + 1) / 2;
    const num_blocks = M * ((K + block_size - 1) / block_size);
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..@intCast(M)] else null;
    tensor_ops_mod.fusedGemvNF4(
        packed_W[0..@intCast(M * k_bytes)],
        scales[0..@intCast(num_blocks)],
        x[0..@intCast(K)],
        bias_slice,
        y[0..@intCast(M)],
        @intCast(M),
        @intCast(K),
        @intCast(block_size),
    );
}

pub export fn hk_fused_gemv_dq8(
    W_i8: [*]const i8,
    scales: [*]const f32,
    x: [*]const f32,
    bias: ?[*]const f32,
    y: [*]f32,
    M: u64,
    K: u64,
    block_size: u32,
) void {
    const num_blocks = M * ((K + block_size - 1) / block_size);
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..@intCast(M)] else null;
    tensor_ops_mod.fusedGemvDQ8(
        W_i8[0..@intCast(M * K)],
        scales[0..@intCast(num_blocks)],
        x[0..@intCast(K)],
        bias_slice,
        y[0..@intCast(M)],
        @intCast(M),
        @intCast(K),
        @intCast(block_size),
    );
}

pub export fn hk_net2wider(
    w_in_old: [*]const f32,
    b_in_old: ?[*]const f32,
    w_in_new: [*]f32,
    b_in_new: ?[*]f32,
    w_out_old: ?[*]const f32,
    w_out_new: ?[*]f32,
    old_out: u64,
    new_out: u64,
    in_f: u64,
    out_f: u64,
    noise_std: f32,
    seed: u64,
) c_int {
    const b_in_old_slice: ?[]const f32 = if (b_in_old) |b| b[0..@intCast(old_out)] else null;
    const b_in_new_slice: ?[]f32 = if (b_in_new) |b| b[0..@intCast(new_out)] else null;
    const w_out_old_slice: ?[]const f32 = if (w_out_old) |w| w[0..@intCast(out_f * old_out)] else null;
    const w_out_new_slice: ?[]f32 = if (w_out_new) |w| w[0..@intCast(out_f * new_out)] else null;

    growth_mod.net2Wider(
        w_in_old[0..@intCast(old_out * in_f)],
        b_in_old_slice,
        w_in_new[0..@intCast(new_out * in_f)],
        b_in_new_slice,
        w_out_old_slice,
        w_out_new_slice,
        @intCast(old_out),
        @intCast(new_out),
        @intCast(in_f),
        @intCast(out_f),
        noise_std,
        seed,
    ) catch return -1;
    return 0;
}

pub export fn hk_net2deeper(weights: [*]f32, bias: ?[*]f32, dim: u64) void {
    const bias_slice: ?[]f32 = if (bias) |b| b[0..@intCast(dim)] else null;
    growth_mod.net2Deeper(weights[0..@intCast(dim * dim)], bias_slice, @intCast(dim));
}

pub export fn hk_net2wider_swiglu(
    w_gate_old: [*]const f32,
    b_gate_old: ?[*]const f32,
    w_up_old: [*]const f32,
    b_up_old: ?[*]const f32,
    w_down_old: [*]const f32,
    b_down_old: ?[*]const f32,
    w_gate_new: [*]f32,
    b_gate_new: ?[*]f32,
    w_up_new: [*]f32,
    b_up_new: ?[*]f32,
    w_down_new: [*]f32,
    b_down_new: ?[*]f32,
    old_inter: u64,
    new_inter: u64,
    in_f: u64,
    out_f: u64,
    zero_init: u8,
    noise_std: f32,
    seed: u64,
) c_int {
    const bg_old: ?[]const f32 = if (b_gate_old) |b| b[0..@intCast(old_inter)] else null;
    const bu_old: ?[]const f32 = if (b_up_old) |b| b[0..@intCast(old_inter)] else null;
    const bd_old: ?[]const f32 = if (b_down_old) |b| b[0..@intCast(out_f)] else null;

    const bg_new: ?[]f32 = if (b_gate_new) |b| b[0..@intCast(new_inter)] else null;
    const bu_new: ?[]f32 = if (b_up_new) |b| b[0..@intCast(new_inter)] else null;
    const bd_new: ?[]f32 = if (b_down_new) |b| b[0..@intCast(out_f)] else null;

    growth_mod.net2WiderSwiGLU(
        w_gate_old[0..@intCast(old_inter * in_f)],
        bg_old,
        w_up_old[0..@intCast(old_inter * in_f)],
        bu_old,
        w_down_old[0..@intCast(out_f * old_inter)],
        bd_old,
        w_gate_new[0..@intCast(new_inter * in_f)],
        bg_new,
        w_up_new[0..@intCast(new_inter * in_f)],
        bu_new,
        w_down_new[0..@intCast(out_f * new_inter)],
        bd_new,
        @intCast(old_inter),
        @intCast(new_inter),
        @intCast(in_f),
        @intCast(out_f),
        (zero_init != 0),
        noise_std,
        seed,
    ) catch return -1;
    return 0;
}

pub export fn hk_expand_vocab(
    embed_old: [*]const f32,
    embed_new: [*]f32,
    lm_head_old: ?[*]const f32,
    lm_head_new: ?[*]f32,
    old_vocab: u64,
    new_vocab: u64,
    hidden_dim: u64,
    seed: u64,
) c_int {
    const head_old: ?[]const f32 = if (lm_head_old) |h| h[0..@intCast(old_vocab * hidden_dim)] else null;
    const head_new: ?[]f32 = if (lm_head_new) |h| h[0..@intCast(new_vocab * hidden_dim)] else null;

    growth_mod.expandVocab(
        embed_old[0..@intCast(old_vocab * hidden_dim)],
        embed_new[0..@intCast(new_vocab * hidden_dim)],
        head_old,
        head_new,
        @intCast(old_vocab),
        @intCast(new_vocab),
        @intCast(hidden_dim),
        seed,
    ) catch return -1;
    return 0;
}

pub export fn hk_plasticity_mask_rows(grad: [*]f32, total_len: u64, cutoff_rows: u64, cols: u64) void {
    growth_mod.applyPlasticityMaskRows(grad[0..@intCast(total_len)], @intCast(cutoff_rows), @intCast(cols));
}

pub export fn hk_plasticity_mask_cols(grad: [*]f32, total_len: u64, num_rows: u64, cutoff_cols: u64, stride_cols: u64) void {
    growth_mod.applyPlasticityMaskCols(grad[0..@intCast(total_len)], @intCast(num_rows), @intCast(cutoff_cols), @intCast(stride_cols));
}

pub export fn hk_forward_swiglu(
    x: [*]const f32,
    w_gate: [*]const f32,
    b_gate: ?[*]const f32,
    w_up: [*]const f32,
    b_up: ?[*]const f32,
    w_down: [*]const f32,
    b_down: ?[*]const f32,
    inter_buf: [*]f32,
    out: [*]f32,
    in_f: u64,
    inter_f: u64,
    out_f: u64,
) c_int {
    const bg: ?[]const f32 = if (b_gate) |b| b[0..@intCast(inter_f)] else null;
    const bu: ?[]const f32 = if (b_up) |b| b[0..@intCast(inter_f)] else null;
    const bd: ?[]const f32 = if (b_down) |b| b[0..@intCast(out_f)] else null;

    tensor_ops_mod.forwardSwiGLUF32(
        x[0..@intCast(in_f)],
        w_gate[0..@intCast(inter_f * in_f)],
        bg,
        w_up[0..@intCast(inter_f * in_f)],
        bu,
        w_down[0..@intCast(out_f * inter_f)],
        bd,
        inter_buf[0..@intCast(inter_f * 2)],
        out[0..@intCast(out_f)],
        @intCast(in_f),
        @intCast(inter_f),
        @intCast(out_f),
    ) catch return -1;
    return 0;
}

pub export fn hk_forward_rmsnorm(
    x: [*]const f32,
    weight: [*]const f32,
    eps: f32,
    out: [*]f32,
    len: u64,
) void {
    tensor_ops_mod.rmsNormF32(
        x[0..@intCast(len)],
        weight[0..@intCast(len)],
        eps,
        out[0..@intCast(len)],
    );
}

pub export fn hk_forward_silu(x: [*]const f32, out: [*]f32, len: u64) void {
    tensor_ops_mod.siluF32(x[0..@intCast(len)], out[0..@intCast(len)]);
}

pub const hk_writer_t = opaque {};

const WriterWrapper = struct {
    arena: std.heap.ArenaAllocator,
    writer: writer_mod.HKWriter,
};

pub export fn hk_writer_create(alignment: u64) ?*hk_writer_t {
    const parent_allocator = std.heap.page_allocator;
    const wrapper = parent_allocator.create(WriterWrapper) catch return null;
    wrapper.arena = std.heap.ArenaAllocator.init(parent_allocator);
    wrapper.writer = writer_mod.HKWriter.init(wrapper.arena.allocator());
    if (alignment > 0) {
        wrapper.writer.setAlignment(@intCast(alignment));
    }
    return @ptrCast(wrapper);
}

pub export fn hk_writer_destroy(writer_ptr: ?*hk_writer_t) void {
    if (writer_ptr == null) return;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    wrapper.arena.deinit();
    std.heap.page_allocator.destroy(wrapper);
}

pub export fn hk_writer_add_metadata_string(writer_ptr: ?*hk_writer_t, key_c: [*:0]const u8, val_c: [*:0]const u8) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const key = std.mem.sliceTo(key_c, 0);
    const val = std.mem.sliceTo(val_c, 0);
    wrapper.writer.addMetadataString(key, val) catch return -1;
    return 0;
}

pub export fn hk_writer_add_metadata_int(writer_ptr: ?*hk_writer_t, key_c: [*:0]const u8, val: i64) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const key = std.mem.sliceTo(key_c, 0);
    wrapper.writer.addMetadataInt(key, val) catch return -1;
    return 0;
}

pub export fn hk_writer_add_metadata_float(writer_ptr: ?*hk_writer_t, key_c: [*:0]const u8, val: f64) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const key = std.mem.sliceTo(key_c, 0);
    wrapper.writer.addMetadataFloat(key, val) catch return -1;
    return 0;
}

pub export fn hk_writer_add_metadata_bool(writer_ptr: ?*hk_writer_t, key_c: [*:0]const u8, val: c_int) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const key = std.mem.sliceTo(key_c, 0);
    wrapper.writer.addMetadataBool(key, val != 0) catch return -1;
    return 0;
}

fn safeStorageType(val: u8) ?format.StorageType {
    inline for (@typeInfo(format.StorageType).@"enum".fields) |f| {
        if (val == f.value) return @enumFromInt(val);
    }
    return null;
}

fn safeTileLayout(val: u8) ?format.TileLayout {
    inline for (@typeInfo(format.TileLayout).@"enum".fields) |f| {
        if (val == f.value) return @enumFromInt(val);
    }
    return null;
}

fn safeSparsityType(val: u8) ?format.SparsityType {
    inline for (@typeInfo(format.SparsityType).@"enum".fields) |f| {
        if (val == f.value) return @enumFromInt(val);
    }
    return null;
}

pub export fn hk_writer_add_tensor(
    writer_ptr: ?*hk_writer_t,
    name_c: [*:0]const u8,
    storage_type_raw: u8,
    tile_layout_raw: u8,
    sparsity_type_raw: u8,
    ndim: u8,
    shape_ptr: [*]const u64,
    data_ptr: [*]const u8,
    data_len: u64,
    sparsity_ratio: f32,
) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const name = std.mem.sliceTo(name_c, 0);
    const alloc = wrapper.arena.allocator();

    const name_dup = alloc.dupe(u8, name) catch return -1;
    const data_dup = alloc.alloc(u8, @intCast(data_len)) catch return -1;
    @memcpy(data_dup, data_ptr[0..@intCast(data_len)]);

    var shape_arr: [format.MAX_DIMS]u64 = [_]u64{0} ** format.MAX_DIMS;
    const count = @min(@as(usize, ndim), format.MAX_DIMS);
    for (0..count) |i| {
        shape_arr[i] = shape_ptr[i];
    }

    const st = safeStorageType(storage_type_raw) orelse return -2;
    const tl = safeTileLayout(tile_layout_raw) orelse return -3;
    const sp = safeSparsityType(sparsity_type_raw) orelse return -4;

    const payload = writer_mod.TensorPayload{
        .name = name_dup,
        .storage_type = st,
        .tile_layout = tl,
        .sparsity_type = sp,
        .ndim = ndim,
        .shape = shape_arr,
        .data = data_dup,
        .sparsity_ratio = sparsity_ratio,
    };

    wrapper.writer.addTensor(payload) catch return -1;
    return 0;
}

pub export fn hk_writer_write_to_file(writer_ptr: ?*hk_writer_t, path_c: [*:0]const u8) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const path = std.mem.sliceTo(path_c, 0);
    wrapper.writer.writeToFile(path) catch return -1;
    return 0;
}

pub export fn hk_writer_set_sharding(writer_ptr: ?*hk_writer_t, split_index: u16, split_count: u16) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    wrapper.writer.setSharding(split_index, split_count);
    return 0;
}

pub export fn hk_reader_is_sharded(reader_ptr: ?*const hk_reader_t) c_int {
    if (reader_ptr == null) return 0;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    return if (wrapper.reader.isSharded()) 1 else 0;
}

pub export fn hk_reader_get_split_index(reader_ptr: ?*const hk_reader_t) u16 {
    if (reader_ptr == null) return 0;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    return wrapper.reader.getSplitIndex();
}

pub export fn hk_reader_get_split_count(reader_ptr: ?*const hk_reader_t) u16 {
    if (reader_ptr == null) return 1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    return wrapper.reader.getSplitCount();
}

pub export fn hk_metadata_patch_in_place(file_path_c: [*:0]const u8, key_c: [*:0]const u8, val_c: [*:0]const u8) c_int {
    const allocator = std.heap.page_allocator;
    const file_path = std.mem.sliceTo(file_path_c, 0);
    const key = std.mem.sliceTo(key_c, 0);
    const val = std.mem.sliceTo(val_c, 0);
    const metadata_mod = @import("metadata.zig");
    metadata_mod.patchFileMetadataInPlace(allocator, file_path, key, val) catch return -1;
    return 0;
}

pub export fn hk_convert_gguf(input_path_c: [*:0]const u8, output_path_c: [*:0]const u8) c_int {
    const allocator = std.heap.page_allocator;
    const input_path = std.mem.sliceTo(input_path_c, 0);
    const output_path = std.mem.sliceTo(output_path_c, 0);
    const gguf_mod = @import("gguf.zig");
    gguf_mod.convertGGUFToHK(input_path, output_path, allocator) catch return -1;
    return 0;
}

pub export fn hk_export_gguf(input_path_c: [*:0]const u8, output_path_c: [*:0]const u8) c_int {
    const allocator = std.heap.page_allocator;
    const input_path = std.mem.sliceTo(input_path_c, 0);
    const output_path = std.mem.sliceTo(output_path_c, 0);
    const gguf_mod = @import("gguf.zig");
    gguf_mod.exportHKToGGUF(input_path, output_path, allocator) catch return -1;
    return 0;
}

pub export fn hk_rope_permute_hf_to_gguf(in: [*]const f32, out: [*]f32, total_len: u64, n_heads: u64, head_dim: u64) void {
    const len: usize = @intCast(total_len);
    tensor_ops_mod.ropePermuteHFToGGUF(in[0..len], out[0..len], @intCast(n_heads), @intCast(head_dim));
}

pub export fn hk_rope_unpermute_gguf_to_hf(in: [*]const f32, out: [*]f32, total_len: u64, n_heads: u64, head_dim: u64) void {
    const len: usize = @intCast(total_len);
    tensor_ops_mod.ropeUnpermuteGGUFToHF(in[0..len], out[0..len], @intCast(n_heads), @intCast(head_dim));
}

pub export fn hk_layernorm_offset_f32(data: [*]f32, len: u64, offset: f32) void {
    tensor_ops_mod.layerNormOffsetF32(data[0..@intCast(len)], offset);
}

pub export fn hk_gemv_q8_0(W: [*]const u8, x: [*]const f32, bias: ?[*]const f32, y: [*]f32, M: u64, K: u64) void {
    const m: usize = @intCast(M);
    const k: usize = @intCast(K);
    const w_bytes_len = m * (k / 32) * @sizeOf(quantization.BlockQ8_0);
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..m] else null;
    tensor_ops_mod.gemvQ8_0(W[0..w_bytes_len], x[0..k], bias_slice, y[0..m], m, k);
}

pub export fn hk_gemv_q4_0(W: [*]const u8, x: [*]const f32, bias: ?[*]const f32, y: [*]f32, M: u64, K: u64) void {
    const m: usize = @intCast(M);
    const k: usize = @intCast(K);
    const w_bytes_len = m * (k / 32) * @sizeOf(quantization.BlockQ4_0);
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..m] else null;
    tensor_ops_mod.gemvQ4_0(W[0..w_bytes_len], x[0..k], bias_slice, y[0..m], m, k);
}

pub export fn hk_gemv_q4_k(W: [*]const u8, x: [*]const f32, bias: ?[*]const f32, y: [*]f32, M: u64, K: u64) void {
    const m: usize = @intCast(M);
    const k: usize = @intCast(K);
    const w_bytes_len = m * (k / 256) * @sizeOf(quantization.BlockQ4_K);
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..m] else null;
    tensor_ops_mod.gemvQ4_K(W[0..w_bytes_len], x[0..k], bias_slice, y[0..m], m, k);
}



