const std = @import("std");
const metadata = @import("metadata.zig");
const format = @import("format.zig");

pub const TokenType = enum(u8) {
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,
};

pub const TokenEntry = struct {
    text: []const u8,
    score: f32 = 0.0,
    token_type: TokenType = .normal,
};

pub const BpePair = struct {
    id1: u32,
    id2: u32,
};

pub const ChatRole = enum {
    system,
    user,
    assistant,
};

pub const ChatMessage = struct {
    role: ChatRole,
    content: []const u8,
};

pub const ChatTemplateKind = enum {
    chatml,
    llama3,
    raw,
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.ArrayList(TokenEntry) = .empty,
    token_to_id: std.StringHashMap(u32),
    bpe_ranks: std.AutoHashMap(BpePair, u32),
    byte_pieces: [256]?u32,

    bos_id: ?u32 = null,
    eos_id: ?u32 = null,
    unk_id: ?u32 = null,
    pad_id: ?u32 = null,
    chat_template: ?[]const u8 = null,

    pub fn init(allocator: std.mem.Allocator) Tokenizer {
        return .{
            .allocator = allocator,
            .vocab = .empty,
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .bpe_ranks = std.AutoHashMap(BpePair, u32).init(allocator),
            .byte_pieces = [_]?u32{null} ** 256,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.vocab.items) |entry| {
            self.allocator.free(entry.text);
        }
        self.vocab.deinit(self.allocator);
        self.token_to_id.deinit();
        self.bpe_ranks.deinit();
        if (self.chat_template) |t| {
            self.allocator.free(t);
        }
    }

    /// Adds a token into the vocabulary. Takes ownership of cloned text string.
    pub fn addToken(self: *Tokenizer, text: []const u8, score: f32, token_type: TokenType) !u32 {
        const id: u32 = @intCast(self.vocab.items.len);
        const text_copy = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(text_copy);

        try self.vocab.append(self.allocator, .{
            .text = text_copy,
            .score = score,
            .token_type = token_type,
        });
        try self.token_to_id.put(text_copy, id);

        // Check for special tokens
        if (std.mem.eql(u8, text, "<s>") or std.mem.eql(u8, text, "<|begin_of_text|>")) {
            self.bos_id = id;
        } else if (std.mem.eql(u8, text, "</s>") or std.mem.eql(u8, text, "<|end_of_text|>") or std.mem.eql(u8, text, "<|im_end|>") or std.mem.eql(u8, text, "<|eot_id|>")) {
            if (self.eos_id == null) self.eos_id = id;
        } else if (std.mem.eql(u8, text, "<unk>")) {
            self.unk_id = id;
        } else if (std.mem.eql(u8, text, "<pad>")) {
            self.pad_id = id;
        }

        // Check for byte token format: <0xXX>
        if (text.len == 6 and std.mem.startsWith(u8, text, "<0x") and text[5] == '>') {
            const hex_part = text[3..5];
            if (std.fmt.parseInt(u8, hex_part, 16)) |b| {
                self.byte_pieces[b] = id;
            } else |_| {}
        } else if (text.len == 1) {
            const b = text[0];
            if (self.byte_pieces[b] == null) {
                self.byte_pieces[b] = id;
            }
        }

        return id;
    }

    /// Adds a BPE merge pair with its rank.
    pub fn addMerge(self: *Tokenizer, first: []const u8, second: []const u8, rank: u32) !void {
        const id1 = self.token_to_id.get(first) orelse return;
        const id2 = self.token_to_id.get(second) orelse return;
        try self.bpe_ranks.put(.{ .id1 = id1, .id2 = id2 }, rank);
    }

    /// Initializes a standard 256-byte fallback vocab for raw byte tokenization when no metadata is available.
    pub fn initByteFallbackVocab(self: *Tokenizer) !void {
        _ = try self.addToken("<pad>", 0.0, .control);
        _ = try self.addToken("<unk>", 0.0, .unknown);
        _ = try self.addToken("<s>", 0.0, .control);
        _ = try self.addToken("</s>", 0.0, .control);

        for (0..256) |b| {
            var buf_byte: [1]u8 = .{@intCast(b)};
            _ = try self.addToken(&buf_byte, 0.0, .byte);
        }
    }

fn parseStringSequence(allocator: std.mem.Allocator, raw: []const u8) !std.ArrayList([]const u8) {
    // 1. Try standard JSON first
    var parsed = std.json.parseFromSlice([]const []const u8, allocator, raw, .{ .ignore_unknown_fields = true }) catch null;
    if (parsed) |*p| {
        defer p.deinit();
        var list: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (list.items) |item| allocator.free(item);
            list.deinit(allocator);
        }
        for (p.value) |s| {
            const dup = try allocator.dupe(u8, s);
            try list.append(allocator, dup);
        }
        return list;
    }

    // 2. Fallback: parse Python-style list repr or quoted strings ['...', '...']
    var list: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (list.items) |item| allocator.free(item);
        list.deinit(allocator);
    }

    var i: usize = 0;
    while (i < raw.len) {
        while (i < raw.len and raw[i] != '\'' and raw[i] != '"') : (i += 1) {}
        if (i >= raw.len) break;
        const quote = raw[i];
        i += 1;
        const start = i;
        while (i < raw.len) {
            if (raw[i] == '\\' and i + 1 < raw.len) {
                i += 2;
            } else if (raw[i] == quote) {
                break;
            } else {
                i += 1;
            }
        }
        const item = try allocator.dupe(u8, raw[start..i]);
        try list.append(allocator, item);
        if (i < raw.len) i += 1;
    }
    return list;
}

    /// Loads tokenizer parameters from an HK MetadataMap.
    pub fn loadFromMetadata(self: *Tokenizer, meta: *const metadata.MetadataMap) !void {
        // 1. Look for tokens JSON
        const tokens_val = meta.get(metadata.StandardKeys.TOKENIZER_TOKENS) orelse
            meta.get("tokenizer.ggml.tokens") orelse
            meta.get("tokens");

        if (tokens_val) |tv| {
            const json_str = switch (tv) {
                .val_string => |s| s,
                .val_json => |j| j,
                else => "",
            };

            if (json_str.len > 0) {
                var tokens_list = parseStringSequence(self.allocator, json_str) catch null;
                if (tokens_list) |*tl| {
                    defer {
                        for (tl.items) |item| self.allocator.free(item);
                        tl.deinit(self.allocator);
                    }
                    for (tl.items) |t_str| {
                        _ = try self.addToken(t_str, 0.0, .normal);
                    }
                }
            }
        }

        // If no tokens loaded, fallback
        if (self.vocab.items.len == 0) {
            try self.initByteFallbackVocab();
            return;
        }

        // 2. Look for merges JSON
        const merges_val = meta.get(metadata.StandardKeys.TOKENIZER_MERGES) orelse
            meta.get("tokenizer.ggml.merges") orelse
            meta.get("merges");

        if (merges_val) |mv| {
            const json_str = switch (mv) {
                .val_string => |s| s,
                .val_json => |j| j,
                else => "",
            };

            if (json_str.len > 0) {
                var merges_list = parseStringSequence(self.allocator, json_str) catch null;
                if (merges_list) |*ml| {
                    defer {
                        for (ml.items) |item| self.allocator.free(item);
                        ml.deinit(self.allocator);
                    }
                    for (ml.items, 0..) |m_str, rank| {
                        var it = std.mem.tokenizeScalar(u8, m_str, ' ');
                        const p1 = it.next();
                        const p2 = it.next();
                        if (p1 != null and p2 != null) {
                            try self.addMerge(p1.?, p2.?, @intCast(rank));
                        }
                    }
                }
            }
        }

        // 3. Override special token IDs if present in metadata
        if (meta.getInt("tokenizer.ggml.bos_token_id")) |bid| {
            self.bos_id = @intCast(bid);
        }
        if (meta.getInt("tokenizer.ggml.eos_token_id")) |eid| {
            self.eos_id = @intCast(eid);
        }
        if (meta.getInt("tokenizer.ggml.unknown_token_id")) |uid| {
            self.unk_id = @intCast(uid);
        }
        if (meta.getInt("tokenizer.ggml.padding_token_id")) |pid| {
            self.pad_id = @intCast(pid);
        }

        // 4. Look for chat template
        if (meta.get(metadata.StandardKeys.TOKENIZER_CHAT_TEMPLATE)) |cv| {
            switch (cv) {
                .val_string => |s| self.chat_template = try self.allocator.dupe(u8, s),
                .val_json => |j| self.chat_template = try self.allocator.dupe(u8, j),
                else => {},
            }
        }
    }

    /// Encodes a raw text string into token IDs.
    pub fn encode(self: *const Tokenizer, text: []const u8, add_bos: bool, add_eos: bool, out_tokens: *std.ArrayList(u32)) !void {
        if (add_bos and self.bos_id != null) {
            try out_tokens.append(self.allocator, self.bos_id.?);
        }

        if (text.len == 0) {
            if (add_eos and self.eos_id != null) {
                try out_tokens.append(self.allocator, self.eos_id.?);
            }
            return;
        }

        // Initial tokenization: maximal prefix or byte fallback
        var temp_ids: std.ArrayList(u32) = .empty;
        defer temp_ids.deinit(self.allocator);

        var i: usize = 0;
        while (i < text.len) {
            var matched: bool = false;
            // Try longest matching prefix up to 32 bytes
            const max_sub_len = @min(text.len - i, 32);
            var len = max_sub_len;
            while (len > 0) : (len -= 1) {
                const sub = text[i .. i + len];
                if (self.token_to_id.get(sub)) |tid| {
                    try temp_ids.append(self.allocator, tid);
                    i += len;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                const b = text[i];
                const tid = self.byte_pieces[b] orelse (self.unk_id orelse 0);
                try temp_ids.append(self.allocator, tid);
                i += 1;
            }
        }

        // BPE Merging Loop if merges exist
        if (self.bpe_ranks.count() > 0 and temp_ids.items.len > 1) {
            while (true) {
                if (temp_ids.items.len < 2) break;

                var best_rank: u32 = std.math.maxInt(u32);
                var best_idx: ?usize = null;

                for (0..temp_ids.items.len - 1) |idx| {
                    const pair = BpePair{
                        .id1 = temp_ids.items[idx],
                        .id2 = temp_ids.items[idx + 1],
                    };
                    if (self.bpe_ranks.get(pair)) |rank| {
                        if (rank < best_rank) {
                            best_rank = rank;
                            best_idx = idx;
                        }
                    }
                }

                if (best_idx == null) break;

                const idx = best_idx.?;
                const id1 = temp_ids.items[idx];
                const id2 = temp_ids.items[idx + 1];

                // Combine token texts
                const t1 = self.vocab.items[id1].text;
                const t2 = self.vocab.items[id2].text;
                var combined: std.ArrayList(u8) = .empty;
                defer combined.deinit(self.allocator);
                try combined.appendSlice(self.allocator, t1);
                try combined.appendSlice(self.allocator, t2);

                if (self.token_to_id.get(combined.items)) |merged_id| {
                    temp_ids.items[idx] = merged_id;
                    _ = temp_ids.orderedRemove(idx + 1);
                } else {
                    break;
                }
            }
        }

        try out_tokens.appendSlice(self.allocator, temp_ids.items);

        if (add_eos and self.eos_id != null) {
            try out_tokens.append(self.allocator, self.eos_id.?);
        }
    }

    /// Decodes a sequence of token IDs back into UTF-8 text.
    pub fn decode(self: *const Tokenizer, token_ids: []const u32, skip_special: bool, out_text: *std.ArrayList(u8)) !void {
        for (token_ids) |tid| {
            if (tid >= self.vocab.items.len) continue;

            const entry = self.vocab.items[tid];
            if (skip_special) {
                if (entry.token_type == .control or entry.token_type == .unknown) {
                    continue;
                }
                if (self.bos_id != null and tid == self.bos_id.?) continue;
                if (self.eos_id != null and tid == self.eos_id.?) continue;
            }

            // Check if it's a byte fallback token: <0xXX>
            if (entry.text.len == 6 and std.mem.startsWith(u8, entry.text, "<0x") and entry.text[5] == '>') {
                const hex_part = entry.text[3..5];
                if (std.fmt.parseInt(u8, hex_part, 16)) |b| {
                    try out_text.append(self.allocator, b);
                    continue;
                } else |_| {}
            }

            try out_text.appendSlice(self.allocator, entry.text);
        }
    }

    /// Formats a conversation into a prompt string.
    pub fn formatChat(
        self: *const Tokenizer,
        kind: ChatTemplateKind,
        messages: []const ChatMessage,
        add_generation_prompt: bool,
        out: *std.ArrayList(u8),
    ) !void {
        switch (kind) {
            .chatml => {
                for (messages) |msg| {
                    const role_str = switch (msg.role) {
                        .system => "system",
                        .user => "user",
                        .assistant => "assistant",
                    };
                    try out.appendSlice(self.allocator, "<|im_start|>");
                    try out.appendSlice(self.allocator, role_str);
                    try out.append(self.allocator, '\n');
                    try out.appendSlice(self.allocator, msg.content);
                    try out.appendSlice(self.allocator, "<|im_end|>\n");
                }
                if (add_generation_prompt) {
                    try out.appendSlice(self.allocator, "<|im_start|>assistant\n");
                }
            },
            .llama3 => {
                try out.appendSlice(self.allocator, "<|begin_of_text|>");
                for (messages) |msg| {
                    const role_str = switch (msg.role) {
                        .system => "system",
                        .user => "user",
                        .assistant => "assistant",
                    };
                    try out.appendSlice(self.allocator, "<|start_header_id|>");
                    try out.appendSlice(self.allocator, role_str);
                    try out.appendSlice(self.allocator, "<|end_header_id|>\n\n");
                    try out.appendSlice(self.allocator, msg.content);
                    try out.appendSlice(self.allocator, "<|eot_id|>");
                }
                if (add_generation_prompt) {
                    try out.appendSlice(self.allocator, "<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
            },
            .raw => {
                for (messages) |msg| {
                    try out.appendSlice(self.allocator, msg.content);
                    try out.append(self.allocator, '\n');
                }
            },
        }
    }
};
