const std = @import("std");

pub const TruncationStrategy = enum(c_int) {
    tail = 0,
    head = 1,
    middle_out = 2,
    sliding_window = 3,

    pub fn fromInt(val: c_int) TruncationStrategy {
        return switch (val) {
            1 => .head,
            2 => .middle_out,
            3 => .sliding_window,
            else => .tail,
        };
    }
};

/// High-performance token truncation in Zig
pub fn truncateTokens(
    tokens: []const u32,
    max_tokens: usize,
    strategy: TruncationStrategy,
    head_ratio: f32,
    out_buf: []u32,
) usize {
    if (tokens.len <= max_tokens) {
        const copy_len = @min(tokens.len, out_buf.len);
        @memcpy(out_buf[0..copy_len], tokens[0..copy_len]);
        return copy_len;
    }

    const limit = @min(max_tokens, out_buf.len);

    switch (strategy) {
        .head => {
            @memcpy(out_buf[0..limit], tokens[0..limit]);
            return limit;
        },
        .tail, .sliding_window => {
            const start = tokens.len - limit;
            @memcpy(out_buf[0..limit], tokens[start..]);
            return limit;
        },
        .middle_out => {
            const ratio = std.math.clamp(head_ratio, 0.0, 1.0);
            var head_len = @as(usize, @intFromFloat(@as(f32, @floatFromInt(limit)) * ratio));
            head_len = @min(head_len, limit);
            const tail_len = limit - head_len;

            // Copy head
            if (head_len > 0) {
                @memcpy(out_buf[0..head_len], tokens[0..head_len]);
            }
            // Copy tail
            if (tail_len > 0) {
                const tail_start = tokens.len - tail_len;
                @memcpy(out_buf[head_len .. head_len + tail_len], tokens[tail_start..]);
            }
            return limit;
        },
    }
}
