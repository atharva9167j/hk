const std = @import("std");
const format = @import("format.zig");

pub const TileDims = struct {
    m: usize,
    k: usize,
};

pub fn getTileDims(layout: format.TileLayout) TileDims {
    return switch (layout) {
        .tile_16x16 => .{ .m = 16, .k = 16 },
        .tile_16x8 => .{ .m = 16, .k = 8 },
        .tile_32x16 => .{ .m = 32, .k = 16 },
        else => .{ .m = 1, .k = 1 },
    };
}

/// Packs a 2D row-major float matrix [M, K] into Tensor-Core tiles.
/// Each tile has dimensions [M_tile, K_tile] with the K-dimension stored contiguously.
/// If M or K are not multiples of the tile dimensions, zero-padding is applied.
pub fn packTilesF32(
    in_row_major: []const f32,
    M: usize,
    K: usize,
    layout: format.TileLayout,
    allocator: std.mem.Allocator,
) ![]f32 {
    const dims = getTileDims(layout);
    if (dims.m <= 1 and dims.k <= 1) {
        const copy = try allocator.alloc(f32, in_row_major.len);
        @memcpy(copy, in_row_major);
        return copy;
    }

    const pad_m = (dims.m - (M % dims.m)) % dims.m;
    const pad_k = (dims.k - (K % dims.k)) % dims.k;
    const padded_m = M + pad_m;
    const padded_k = K + pad_k;

    const num_tiles_m = padded_m / dims.m;
    const num_tiles_k = padded_k / dims.k;
    const total_elements = num_tiles_m * num_tiles_k * dims.m * dims.k;

    const tiled_out = try allocator.alloc(f32, total_elements);
    @memset(tiled_out, 0.0);

    var tile_idx: usize = 0;
    for (0..num_tiles_m) |tm| {
        for (0..num_tiles_k) |tk| {
            const tile_start = tile_idx * dims.m * dims.k;
            for (0..dims.m) |r| {
                const global_r = tm * dims.m + r;
                for (0..dims.k) |c| {
                    const global_c = tk * dims.k + c;
                    const val = if (global_r < M and global_c < K)
                        in_row_major[global_r * K + global_c]
                    else
                        0.0;
                    tiled_out[tile_start + (r * dims.k + c)] = val;
                }
            }
            tile_idx += 1;
        }
    }

    return tiled_out;
}

/// Unpacks Tensor-Core tiled float matrix back into standard row-major [M, K].
pub fn unpackTilesF32(
    in_tiled: []const f32,
    M: usize,
    K: usize,
    layout: format.TileLayout,
    out_row_major: []f32,
) !void {
    const dims = getTileDims(layout);
    if (dims.m <= 1 and dims.k <= 1) {
        const count = @min(in_tiled.len, out_row_major.len);
        @memcpy(out_row_major[0..count], in_tiled[0..count]);
        return;
    }

    const pad_m = (dims.m - (M % dims.m)) % dims.m;
    const pad_k = (dims.k - (K % dims.k)) % dims.k;
    const padded_m = M + pad_m;
    const padded_k = K + pad_k;

    const num_tiles_m = padded_m / dims.m;
    const num_tiles_k = padded_k / dims.k;

    var tile_idx: usize = 0;
    for (0..num_tiles_m) |tm| {
        for (0..num_tiles_k) |tk| {
            const tile_start = tile_idx * dims.m * dims.k;
            for (0..dims.m) |r| {
                const global_r = tm * dims.m + r;
                for (0..dims.k) |c| {
                    const global_c = tk * dims.k + c;
                    if (global_r < M and global_c < K) {
                        const val = in_tiled[tile_start + (r * dims.k + c)];
                        out_row_major[global_r * K + global_c] = val;
                    }
                }
            }
            tile_idx += 1;
        }
    }
}
