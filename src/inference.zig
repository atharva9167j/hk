const std = @import("std");
const format = @import("format.zig");
const reader_mod = @import("reader.zig");
const metadata_mod = @import("metadata.zig");
const tensor_ops = @import("tensor_ops.zig");
const quantization = @import("quantization.zig");

pub const ModelConfig = struct {
    dim: usize = 512,
    hidden_dim: usize = 1376,
    n_layers: usize = 4,
    n_heads: usize = 8,
    n_kv_heads: usize = 4,
    vocab_size: usize = 32000,
    max_seq_len: usize = 2048,
    head_dim: usize = 64,
    norm_eps: f32 = 1e-5,
    rope_theta: f32 = 10000.0,
};

pub const TensorRef = struct {
    data: []const u8,
    storage_type: format.StorageType,
    rows: usize,
    cols: usize,
};

pub const LayerWeights = struct {
    attn_norm: []const f32,
    wq: TensorRef,
    wk: TensorRef,
    wv: TensorRef,
    wo: TensorRef,
    ffn_norm: []const f32,
    w_gate: TensorRef,
    w_up: TensorRef,
    w_down: TensorRef,
};

pub const KVCache = struct {
    allocator: std.mem.Allocator,
    key_cache: []f32,
    val_cache: []f32,
    max_seq_len: usize,
    n_layers: usize,
    kv_dim: usize,

    pub fn init(allocator: std.mem.Allocator, n_layers: usize, max_seq_len: usize, kv_dim: usize) !KVCache {
        const total_elements = n_layers * max_seq_len * kv_dim;
        const key_cache = try allocator.alloc(f32, total_elements);
        errdefer allocator.free(key_cache);
        const val_cache = try allocator.alloc(f32, total_elements);
        errdefer allocator.free(val_cache);

        @memset(key_cache, 0.0);
        @memset(val_cache, 0.0);

        return .{
            .allocator = allocator,
            .key_cache = key_cache,
            .val_cache = val_cache,
            .max_seq_len = max_seq_len,
            .n_layers = n_layers,
            .kv_dim = kv_dim,
        };
    }

    pub fn deinit(self: *KVCache) void {
        self.allocator.free(self.key_cache);
        self.allocator.free(self.val_cache);
    }

    pub fn reset(self: *KVCache) void {
        @memset(self.key_cache, 0.0);
        @memset(self.val_cache, 0.0);
    }

    pub fn getKeySlice(self: *KVCache, layer: usize, pos: usize) []f32 {
        const safe_pos = @min(pos, self.max_seq_len - 1);
        const offset = (layer * self.max_seq_len + safe_pos) * self.kv_dim;
        return self.key_cache[offset .. offset + self.kv_dim];
    }

    pub fn getValSlice(self: *KVCache, layer: usize, pos: usize) []f32 {
        const safe_pos = @min(pos, self.max_seq_len - 1);
        const offset = (layer * self.max_seq_len + safe_pos) * self.kv_dim;
        return self.val_cache[offset .. offset + self.kv_dim];
    }
};

pub const TransformerEngine = struct {
    allocator: std.mem.Allocator,
    config: ModelConfig,
    tok_embeddings: TensorRef,
    layers: std.ArrayList(LayerWeights) = .empty,
    output_norm: []const f32,
    lm_head: TensorRef,
    cache: KVCache,

    // Scratch working buffers
    x: []f32,
    xb: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    att: []f32,
    gate: []f32,
    up: []f32,
    logits: []f32,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, tok_embeddings: TensorRef, output_norm: []const f32, lm_head: TensorRef) !TransformerEngine {
        const kv_dim = @max(config.n_kv_heads * config.head_dim, config.dim);
        var cache = try KVCache.init(allocator, config.n_layers, config.max_seq_len, kv_dim);
        errdefer cache.deinit();

        const x = try allocator.alloc(f32, config.dim);
        errdefer allocator.free(x);
        const xb = try allocator.alloc(f32, config.dim);
        errdefer allocator.free(xb);
        const q = try allocator.alloc(f32, @max(config.n_heads * config.head_dim, config.dim));
        errdefer allocator.free(q);
        const k = try allocator.alloc(f32, kv_dim);
        errdefer allocator.free(k);
        const v = try allocator.alloc(f32, kv_dim);
        errdefer allocator.free(v);
        const att = try allocator.alloc(f32, config.max_seq_len);
        errdefer allocator.free(att);
        const gate = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(gate);
        const up = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(up);
        const logits = try allocator.alloc(f32, config.vocab_size);
        errdefer allocator.free(logits);

        return .{
            .allocator = allocator,
            .config = config,
            .tok_embeddings = tok_embeddings,
            .layers = .empty,
            .output_norm = output_norm,
            .lm_head = lm_head,
            .cache = cache,
            .x = x,
            .xb = xb,
            .q = q,
            .k = k,
            .v = v,
            .att = att,
            .gate = gate,
            .up = up,
            .logits = logits,
        };
    }

    /// Constructs and initializes a TransformerEngine from an open HKReader
    pub fn initFromReader(allocator: std.mem.Allocator, reader: *reader_mod.HKReader) !*TransformerEngine {
        var cfg = ModelConfig{};

        if (reader.metadata_map.findInt(metadata_mod.StandardKeys.ATTN_HEAD_COUNT) orelse reader.metadata_map.findInt("head_count")) |v| {
            if (v > 0) cfg.n_heads = @intCast(v);
        }
        if (reader.metadata_map.findInt(metadata_mod.StandardKeys.ATTN_HEAD_COUNT_KV) orelse reader.metadata_map.findInt("head_count_kv")) |v| {
            if (v > 0) cfg.n_kv_heads = @intCast(v);
        } else {
            cfg.n_kv_heads = cfg.n_heads;
        }
        if (reader.metadata_map.findFloat(metadata_mod.StandardKeys.ATTN_LAYER_NORM_RMS_EPS) orelse reader.metadata_map.findFloat("norm_eps")) |v| {
            if (v > 0.0) cfg.norm_eps = @floatCast(v);
        }
        if (reader.metadata_map.findFloat(metadata_mod.StandardKeys.ROPE_FREQ_BASE) orelse reader.metadata_map.findFloat("rope_theta")) |v| {
            if (v > 0.0) cfg.rope_theta = @floatCast(v);
        }
        if (reader.metadata_map.findInt("context_length") orelse reader.metadata_map.findInt("max_position_embeddings") orelse reader.metadata_map.findInt("max_seq_len")) |v| {
            if (v > 0) cfg.max_seq_len = @intCast(v);
        }

        // Find token_embd.weight
        const emb_entry = reader.toc.find("token_embd.weight") orelse
            reader.toc.find("model.embed_tokens.weight") orelse
            return error.MissingEmbeddingTensor;

        cfg.vocab_size = @intCast(emb_entry.shape[0]);
        cfg.dim = @intCast(emb_entry.shape[1]);

        if (reader.metadata_map.findInt(metadata_mod.StandardKeys.ROPE_DIMENSION_COUNT) orelse reader.metadata_map.findInt("head_dim")) |v| {
            if (v > 0) cfg.head_dim = @intCast(v);
        } else {
            if (cfg.n_heads > 0) {
                cfg.head_dim = cfg.dim / cfg.n_heads;
            } else {
                cfg.head_dim = cfg.dim;
            }
        }

        // Detect or refine n_heads and n_kv_heads directly from layer 0 projection shapes if available
        const wq_0_entry = reader.toc.find("blk.0.attn_q.weight") orelse
            reader.toc.find("model.layers.0.self_attn.q_proj.weight");
        if (wq_0_entry) |wq_e| {
            const wq_rows: usize = @intCast(wq_e.shape[0]);
            if (cfg.head_dim > 0 and wq_rows > 0) {
                cfg.n_heads = @max(1, wq_rows / cfg.head_dim);
            }
        }

        const wk_0_entry = reader.toc.find("blk.0.attn_k.weight") orelse
            reader.toc.find("model.layers.0.self_attn.k_proj.weight");
        if (wk_0_entry) |wk_e| {
            const wk_rows: usize = @intCast(wk_e.shape[0]);
            if (cfg.head_dim > 0 and wk_rows > 0) {
                cfg.n_kv_heads = @max(1, wk_rows / cfg.head_dim);
            }
        }

        const emb_data = try reader.getTensorData(emb_entry);
        const tok_embeddings = TensorRef{
            .data = emb_data,
            .storage_type = emb_entry.storage_type,
            .rows = cfg.vocab_size,
            .cols = cfg.dim,
        };

        // Find output_norm.weight - safely copy to owned aligned buffer
        const norm_entry = reader.toc.find("output_norm.weight") orelse
            reader.toc.find("model.norm.weight") orelse
            return error.MissingOutputNormTensor;
        const norm_data = try reader.getTensorData(norm_entry);
        const output_norm: [*]const f32 = @ptrCast(@alignCast(norm_data.ptr));

        // Find lm_head / output.weight
        const head_entry_opt = reader.toc.find("output.weight") orelse
            reader.toc.find("lm_head.weight");
        const lm_head = if (head_entry_opt) |h_entry| blk: {
            const h_data = try reader.getTensorData(h_entry);
            break :blk TensorRef{
                .data = h_data,
                .storage_type = h_entry.storage_type,
                .rows = @intCast(h_entry.shape[0]),
                .cols = @intCast(h_entry.shape[1]),
            };
        } else tok_embeddings;

        // Detect number of layers by checking blk.0, blk.1, ...
        var layer_count: usize = 0;
        var name_buf: [64]u8 = undefined;
        while (true) : (layer_count += 1) {
            const test_name = std.fmt.bufPrint(&name_buf, "blk.{}.attn_q.weight", .{layer_count}) catch break;
            if (reader.toc.find(test_name) == null) {
                const alt_name = std.fmt.bufPrint(&name_buf, "model.layers.{}.self_attn.q_proj.weight", .{layer_count}) catch break;
                if (reader.toc.find(alt_name) == null) break;
            }
        }
        if (layer_count == 0) return error.NoLayersFound;
        cfg.n_layers = layer_count;

        // Determine intermediate dimension from blk.0.ffn_gate.weight
        const gate_0_entry = reader.toc.find("blk.0.ffn_gate.weight") orelse
            reader.toc.find("model.layers.0.mlp.gate_proj.weight") orelse
            return error.MissingFFNGateTensor;
        cfg.hidden_dim = @intCast(gate_0_entry.shape[0]);

        const engine = try allocator.create(TransformerEngine);
        errdefer allocator.destroy(engine);

        engine.* = try TransformerEngine.init(allocator, cfg, tok_embeddings, output_norm[0..cfg.dim], lm_head);
        errdefer engine.deinit();

        for (0..cfg.n_layers) |l| {
            var buf1: [64]u8 = undefined;
            var buf2: [64]u8 = undefined;
            var buf3: [64]u8 = undefined;
            var buf4: [64]u8 = undefined;
            var buf5: [64]u8 = undefined;
            var buf6: [64]u8 = undefined;
            var buf7: [64]u8 = undefined;
            var buf8: [64]u8 = undefined;
            var buf9: [64]u8 = undefined;

            const attn_norm_name = std.fmt.bufPrint(&buf1, "blk.{}.attn_norm.weight", .{l}) catch return error.NameTooLong;
            const wq_name = std.fmt.bufPrint(&buf2, "blk.{}.attn_q.weight", .{l}) catch return error.NameTooLong;
            const wk_name = std.fmt.bufPrint(&buf3, "blk.{}.attn_k.weight", .{l}) catch return error.NameTooLong;
            const wv_name = std.fmt.bufPrint(&buf4, "blk.{}.attn_v.weight", .{l}) catch return error.NameTooLong;
            const wo_name = std.fmt.bufPrint(&buf5, "blk.{}.attn_output.weight", .{l}) catch return error.NameTooLong;
            const ffn_norm_name = std.fmt.bufPrint(&buf6, "blk.{}.ffn_norm.weight", .{l}) catch return error.NameTooLong;
            const wgate_name = std.fmt.bufPrint(&buf7, "blk.{}.ffn_gate.weight", .{l}) catch return error.NameTooLong;
            const wup_name = std.fmt.bufPrint(&buf8, "blk.{}.ffn_up.weight", .{l}) catch return error.NameTooLong;
            const wdown_name = std.fmt.bufPrint(&buf9, "blk.{}.ffn_down.weight", .{l}) catch return error.NameTooLong;

            const attn_norm_e = reader.toc.find(attn_norm_name) orelse return error.MissingLayerTensor;
            const wq_e = reader.toc.find(wq_name) orelse return error.MissingLayerTensor;
            const wk_e = reader.toc.find(wk_name) orelse return error.MissingLayerTensor;
            const wv_e = reader.toc.find(wv_name) orelse return error.MissingLayerTensor;
            const wo_e = reader.toc.find(wo_name) orelse return error.MissingLayerTensor;
            const ffn_norm_e = reader.toc.find(ffn_norm_name) orelse return error.MissingLayerTensor;
            const wgate_e = reader.toc.find(wgate_name) orelse return error.MissingLayerTensor;
            const wup_e = reader.toc.find(wup_name) orelse return error.MissingLayerTensor;
            const wdown_e = reader.toc.find(wdown_name) orelse return error.MissingLayerTensor;

            const attn_norm_data = try reader.getTensorData(attn_norm_e);
            const ffn_norm_data = try reader.getTensorData(ffn_norm_e);

            const a_norm_f32: [*]const f32 = @ptrCast(@alignCast(attn_norm_data.ptr));
            const f_norm_f32: [*]const f32 = @ptrCast(@alignCast(ffn_norm_data.ptr));

            try engine.layers.append(allocator, .{
                .attn_norm = a_norm_f32[0..cfg.dim],
                .wq = .{ .data = try reader.getTensorData(wq_e), .storage_type = wq_e.storage_type, .rows = @intCast(wq_e.shape[0]), .cols = @intCast(wq_e.shape[1]) },
                .wk = .{ .data = try reader.getTensorData(wk_e), .storage_type = wk_e.storage_type, .rows = @intCast(wk_e.shape[0]), .cols = @intCast(wk_e.shape[1]) },
                .wv = .{ .data = try reader.getTensorData(wv_e), .storage_type = wv_e.storage_type, .rows = @intCast(wv_e.shape[0]), .cols = @intCast(wv_e.shape[1]) },
                .wo = .{ .data = try reader.getTensorData(wo_e), .storage_type = wo_e.storage_type, .rows = @intCast(wo_e.shape[0]), .cols = @intCast(wo_e.shape[1]) },
                .ffn_norm = f_norm_f32[0..cfg.dim],
                .w_gate = .{ .data = try reader.getTensorData(wgate_e), .storage_type = wgate_e.storage_type, .rows = @intCast(wgate_e.shape[0]), .cols = @intCast(wgate_e.shape[1]) },
                .w_up = .{ .data = try reader.getTensorData(wup_e), .storage_type = wup_e.storage_type, .rows = @intCast(wup_e.shape[0]), .cols = @intCast(wup_e.shape[1]) },
                .w_down = .{ .data = try reader.getTensorData(wdown_e), .storage_type = wdown_e.storage_type, .rows = @intCast(wdown_e.shape[0]), .cols = @intCast(wdown_e.shape[1]) },
            });
        }

        return engine;
    }

    pub fn deinit(self: *TransformerEngine) void {
        self.layers.deinit(self.allocator);
        self.cache.deinit();
        self.allocator.free(self.x);
        self.allocator.free(self.xb);
        self.allocator.free(self.q);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.allocator.free(self.att);
        self.allocator.free(self.gate);
        self.allocator.free(self.up);
        self.allocator.free(self.logits);
    }

    /// Linear matrix-vector multiplication dispatching to quantized SIMD kernels
    pub fn matVec(weight: TensorRef, in_x: []const f32, bias: ?[]const f32, out_y: []f32) void {
        const safe_rows = @min(weight.rows, out_y.len);
        switch (weight.storage_type) {
            .q8_0 => tensor_ops.gemvQ8_0(weight.data, in_x, bias, out_y[0..safe_rows], safe_rows, weight.cols),
            .q4_0 => tensor_ops.gemvQ4_0(weight.data, in_x, bias, out_y[0..safe_rows], safe_rows, weight.cols),
            .q4_k => tensor_ops.gemvQ4_K(weight.data, in_x, bias, out_y[0..safe_rows], safe_rows, weight.cols),
            .f32 => {
                if (std.mem.isAligned(@intFromPtr(weight.data.ptr), @alignOf(f32))) {
                    const w_f32: [*]const f32 = @ptrCast(@alignCast(weight.data.ptr));
                    tensor_ops.gemvF32(w_f32[0 .. weight.rows * weight.cols], in_x, bias, out_y[0..safe_rows], safe_rows, weight.cols);
                } else {
                    for (0..safe_rows) |r| {
                        var dot: f32 = 0.0;
                        const row_offset = r * weight.cols * 4;
                        for (0..weight.cols) |c| {
                            const float_bytes = weight.data[row_offset + c * 4 .. row_offset + (c + 1) * 4];
                            const f_val: f32 = @bitCast(std.mem.readInt(u32, float_bytes[0..4], .little));
                            dot += f_val * in_x[c];
                        }
                        if (bias) |b| {
                            if (r < b.len) dot += b[r];
                        }
                        out_y[r] = dot;
                    }
                }
            },
            else => {
                if (std.mem.isAligned(@intFromPtr(weight.data.ptr), @alignOf(f32))) {
                    const w_f32: [*]const f32 = @ptrCast(@alignCast(weight.data.ptr));
                    tensor_ops.gemvF32(w_f32[0 .. weight.rows * weight.cols], in_x, bias, out_y[0..safe_rows], safe_rows, weight.cols);
                } else {
                    for (0..safe_rows) |r| {
                        var dot: f32 = 0.0;
                        const row_offset = r * weight.cols * 4;
                        for (0..weight.cols) |c| {
                            const float_bytes = weight.data[row_offset + c * 4 .. row_offset + (c + 1) * 4];
                            const f_val: f32 = @bitCast(std.mem.readInt(u32, float_bytes[0..4], .little));
                            dot += f_val * in_x[c];
                        }
                        if (bias) |b| {
                            if (r < b.len) dot += b[r];
                        }
                        out_y[r] = dot;
                    }
                }
            },
        }
    }

    /// Embedding lookup for token ID
    pub fn lookupEmbedding(self: *const TransformerEngine, token: u32, out_x: []f32) void {
        const dim = self.config.dim;
        const tid = if (token < self.config.vocab_size) token else 0;

        switch (self.tok_embeddings.storage_type) {
            .f32 => {
                if (std.mem.isAligned(@intFromPtr(self.tok_embeddings.data.ptr), @alignOf(f32))) {
                    const w_f32: [*]const f32 = @ptrCast(@alignCast(self.tok_embeddings.data.ptr));
                    const row = w_f32[tid * dim .. (tid + 1) * dim];
                    @memcpy(out_x[0..dim], row);
                } else {
                    for (0..dim) |i| {
                        const off = (tid * dim + i) * 4;
                        const float_bytes = self.tok_embeddings.data[off .. off + 4];
                        out_x[i] = @bitCast(std.mem.readInt(u32, float_bytes[0..4], .little));
                    }
                }
            },
            .q8_0 => {
                const blocks_per_row = dim / 32;
                const row_bytes = blocks_per_row * 34;
                const row_slice = self.tok_embeddings.data[tid * row_bytes .. (tid + 1) * row_bytes];
                for (0..blocks_per_row) |b| {
                    const blk_bytes = row_slice[b * 34 .. (b + 1) * 34];
                    const d_raw = std.mem.readInt(u16, blk_bytes[0..2], .little);
                    const d: f32 = @floatCast(@as(f16, @bitCast(d_raw)));
                    for (0..32) |i| {
                        const q: i8 = @bitCast(blk_bytes[2 + i]);
                        out_x[b * 32 + i] = @as(f32, @floatFromInt(q)) * d;
                    }
                }
            },
            .q4_0 => {
                const blocks_per_row = dim / 32;
                const row_bytes = blocks_per_row * 18;
                const row_slice = self.tok_embeddings.data[tid * row_bytes .. (tid + 1) * row_bytes];
                for (0..blocks_per_row) |b| {
                    const blk_bytes = row_slice[b * 18 .. (b + 1) * 18];
                    const d_raw = std.mem.readInt(u16, blk_bytes[0..2], .little);
                    const d: f32 = @floatCast(@as(f16, @bitCast(d_raw)));
                    for (0..16) |i| {
                        const byte = blk_bytes[2 + i];
                        const q0: i8 = @as(i8, @intCast(byte & 0x0F)) - 8;
                        const q1: i8 = @as(i8, @intCast((byte >> 4) & 0x0F)) - 8;
                        out_x[b * 32 + i] = @as(f32, @floatFromInt(q0)) * d;
                        out_x[b * 32 + i + 16] = @as(f32, @floatFromInt(q1)) * d;
                    }
                }
            },
            else => {
                @memset(out_x[0..dim], 0.0);
            },
        }
    }

    /// Single autoregressive forward pass step: (token, pos) -> logits
    pub fn forward(self: *TransformerEngine, token: u32, pos: usize) []const f32 {
        const cfg = self.config;
        const dim = cfg.dim;
        const head_dim = cfg.head_dim;
        const n_heads = cfg.n_heads;
        const n_kv_heads = cfg.n_kv_heads;
        const n_rep = if (n_kv_heads > 0) @max(1, n_heads / n_kv_heads) else 1;

        // 1. Embedding lookup
        self.lookupEmbedding(token, self.x);

        // 2. Transformer layers
        for (self.layers.items, 0..) |layer, l| {
            // Attention RMSNorm
            tensor_ops.rmsNormF32(self.x, layer.attn_norm, cfg.norm_eps, self.xb);

            // Compute Q, K, V projections
            matVec(layer.wq, self.xb, null, self.q);
            matVec(layer.wk, self.xb, null, self.k);
            matVec(layer.wv, self.xb, null, self.v);

            // RoPE Rotary Position Embedding
            if (head_dim >= 2) {
                const half_dim = head_dim / 2;
                for (0..n_heads) |h| {
                    if ((h + 1) * head_dim > self.q.len) break;
                    const q_head = self.q[h * head_dim .. (h + 1) * head_dim];
                    for (0..half_dim) |i| {
                        const freq = 1.0 / std.math.pow(f32, cfg.rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
                        const val = @as(f32, @floatFromInt(pos)) * freq;
                        const cos_val = @cos(val);
                        const sin_val = @sin(val);
                        const v0 = q_head[2 * i];
                        const v1 = q_head[2 * i + 1];
                        q_head[2 * i] = v0 * cos_val - v1 * sin_val;
                        q_head[2 * i + 1] = v0 * sin_val + v1 * cos_val;
                    }
                }

                for (0..n_kv_heads) |h| {
                    if ((h + 1) * head_dim > self.k.len) break;
                    const k_head = self.k[h * head_dim .. (h + 1) * head_dim];
                    for (0..half_dim) |i| {
                        const freq = 1.0 / std.math.pow(f32, cfg.rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
                        const val = @as(f32, @floatFromInt(pos)) * freq;
                        const cos_val = @cos(val);
                        const sin_val = @sin(val);
                        const v0 = k_head[2 * i];
                        const v1 = k_head[2 * i + 1];
                        k_head[2 * i] = v0 * cos_val - v1 * sin_val;
                        k_head[2 * i + 1] = v0 * sin_val + v1 * cos_val;
                    }
                }
            }

            // Save key and value into KV cache
            const k_cached = self.cache.getKeySlice(l, pos);
            const v_cached = self.cache.getValSlice(l, pos);
            const kv_copy_len = @min(k_cached.len, self.k.len);
            @memcpy(k_cached[0..kv_copy_len], self.k[0..kv_copy_len]);
            @memcpy(v_cached[0..kv_copy_len], self.v[0..kv_copy_len]);

            // Grouped-Query Multi-Head Attention
            @memset(self.xb, 0.0);
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(@max(head_dim, 1))));

            for (0..n_heads) |h| {
                if ((h + 1) * head_dim > self.q.len or (h + 1) * head_dim > self.xb.len) break;
                const q_head = self.q[h * head_dim .. (h + 1) * head_dim];
                const kv_head_idx = h / n_rep;

                // Compute attention scores for all cached tokens 0..=pos
                const max_t = @min(pos + 1, self.config.max_seq_len);
                for (0..max_t) |t| {
                    const k_t = self.cache.getKeySlice(l, t);
                    const k_head = k_t[kv_head_idx * head_dim .. (kv_head_idx + 1) * head_dim];
                    self.att[t] = tensor_ops.dotProductF32(q_head, k_head) * scale;
                }

                // Softmax over 0..=pos
                tensor_ops.softmaxF32(self.att[0..max_t], self.att[0..max_t]);

                // Weighted sum over V
                const out_head = self.xb[h * head_dim .. (h + 1) * head_dim];
                for (0..max_t) |t| {
                    const a = self.att[t];
                    const v_t = self.cache.getValSlice(l, t);
                    const v_head = v_t[kv_head_idx * head_dim .. (kv_head_idx + 1) * head_dim];
                    for (0..head_dim) |d| {
                        out_head[d] += a * v_head[d];
                    }
                }
            }

            // Attention Output Projection: wo * xb -> q
            matVec(layer.wo, self.xb, null, self.q);

            // Residual connection: x = x + q
            for (0..dim) |i| {
                self.x[i] += self.q[i];
            }

            // Feed-Forward SwiGLU Block
            tensor_ops.rmsNormF32(self.x, layer.ffn_norm, cfg.norm_eps, self.xb);
            matVec(layer.w_gate, self.xb, null, self.gate);
            matVec(layer.w_up, self.xb, null, self.up);

            // SiLU(gate) * up
            for (0..cfg.hidden_dim) |i| {
                const g = self.gate[i];
                const silu_g = g / (1.0 + @exp(-g));
                self.gate[i] = silu_g * self.up[i];
            }

            // Down projection: w_down * gate -> xb
            matVec(layer.w_down, self.gate, null, self.xb);

            // Residual connection: x = x + xb
            for (0..dim) |i| {
                self.x[i] += self.xb[i];
            }
        }

        // 3. Output RMSNorm
        tensor_ops.rmsNormF32(self.x, self.output_norm, cfg.norm_eps, self.xb);

        // 4. LM Head projection -> logits
        matVec(self.lm_head, self.xb, null, self.logits);

        return self.logits;
    }
};
