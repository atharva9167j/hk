const std = @import("std");
const growth = @import("growth.zig");

pub const GrowthGovernor = struct {
    max_vram_mb: u64,
    max_growth_ratio: f32,

    pub fn init(max_vram_mb: u64, max_growth_ratio: f32) GrowthGovernor {
        return .{
            .max_vram_mb = max_vram_mb,
            .max_growth_ratio = max_growth_ratio,
        };
    }

    pub fn canGrow(
        self: GrowthGovernor,
        current_params: u64,
        added_params: u64,
        dtype_bytes: u32,
        out_reason: []u8,
    ) struct { approved: bool, reason: []const u8 } {
        const total_params = current_params + added_params;
        const cur_f = @as(f32, @floatFromInt(current_params));
        const tot_f = @as(f32, @floatFromInt(total_params));
        const ratio = if (cur_f > 0) tot_f / cur_f else 1.0;

        if (ratio > self.max_growth_ratio) {
            const msg = std.fmt.bufPrint(out_reason, "Growth ratio {d:.2}x exceeds max limit {d:.2}x", .{ ratio, self.max_growth_ratio }) catch "Growth ratio exceeds limit";
            return .{ .approved = false, .reason = msg };
        }

        const added_bytes = added_params * dtype_bytes;
        const added_mb = added_bytes / (1024 * 1024);
        if (added_mb > self.max_vram_mb) {
            const msg = std.fmt.bufPrint(out_reason, "Memory requirement {d} MB exceeds budget {d} MB", .{ added_mb, self.max_vram_mb }) catch "Memory exceeds budget";
            return .{ .approved = false, .reason = msg };
        }

        const msg = std.fmt.bufPrint(out_reason, "Approved", .{}) catch "Approved";
        return .{ .approved = true, .reason = msg };
    }
};

/// Expands vocabulary embeddings [V, D] -> [V', D] with exact function preservation on 0..V-1
pub fn expandVocabEmbeddings(
    old_embed: []const f32,
    old_vocab: usize,
    hidden_size: usize,
    new_vocab: usize,
    new_embed: []f32,
    init_std: f32,
    seed: u64,
) void {
    if (new_vocab <= old_vocab) return;

    // 1. Copy old embeddings exactly
    const old_elements = old_vocab * hidden_size;
    @memcpy(new_embed[0..old_elements], old_embed[0..old_elements]);

    // 2. Initialize newly added tokens with normal distribution
    var prng = growth.Prng.init(seed);
    const added_tokens = new_vocab - old_vocab;
    const added_elements = added_tokens * hidden_size;
    const new_slice = new_embed[old_elements .. old_elements + added_elements];

    for (0..added_elements) |i| {
        new_slice[i] = prng.nextGaussian(init_std);
    }
}

/// Initializes plasticity shields / capacity preservation gradient masks
pub fn initPlasticityMask(
    mask: []f32,
    total_units: usize,
    base_units: usize,
    decay_rate: f32,
) void {
    const n = @min(mask.len, total_units);
    for (0..n) |i| {
        if (i < base_units) {
            // Protect base units with lower learning rate factor (plasticity shield)
            mask[i] = decay_rate;
        } else {
            // Full plasticity for expanded units
            mask[i] = 1.0;
        }
    }
}
