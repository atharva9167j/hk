const std = @import("std");
const format = @import("format.zig");
const reader_mod = @import("reader.zig");
const metadata_mod = @import("metadata.zig");
const tensor_ops = @import("tensor_ops.zig");
const quantization = @import("quantization.zig");
const cuda = @import("cuda.zig");
const graph_mod = @import("graph.zig");

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
    owns_norms: bool = false,
    attn_q_norm: ?[]const f32 = null,
    attn_k_norm: ?[]const f32 = null,

    // GPU offload handles (populated when offloaded to device):
    gpu_wq: ?cuda.DeviceBuffer = null,
    gpu_wk: ?cuda.DeviceBuffer = null,
    gpu_wv: ?cuda.DeviceBuffer = null,
    gpu_wo: ?cuda.DeviceBuffer = null,
    gpu_w_gate: ?cuda.DeviceBuffer = null,
    gpu_w_up: ?cuda.DeviceBuffer = null,
    gpu_w_down: ?cuda.DeviceBuffer = null,
    gpu_attn_norm: ?cuda.DeviceBuffer = null,
    gpu_ffn_norm: ?cuda.DeviceBuffer = null,
    gpu_attn_q_norm: ?cuda.DeviceBuffer = null,
    gpu_attn_k_norm: ?cuda.DeviceBuffer = null,
    is_gpu: bool = false,
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

pub const EngineOptions = struct {
    gpu_layers: ?usize = null, // null = auto (full offload if CUDA available, else 0)
};

pub const TransformerEngine = struct {
    allocator: std.mem.Allocator,
    config: ModelConfig,
    tok_embeddings: TensorRef,
    layers: std.ArrayList(LayerWeights) = .empty,
    output_norm: []const f32,
    owns_output_norm: bool = false,
    lm_head: TensorRef,
    cache: KVCache,

    // Dynamic offloading controls
    n_gpu_layers: usize = 0,

    // Device scratch buffers (allocated if n_gpu_layers > 0 and CUDA is available)
    d_x: ?cuda.DeviceBuffer = null,
    d_xb: ?cuda.DeviceBuffer = null,
    d_q: ?cuda.DeviceBuffer = null,
    d_k: ?cuda.DeviceBuffer = null,
    d_v: ?cuda.DeviceBuffer = null,
    d_gate: ?cuda.DeviceBuffer = null,
    d_up: ?cuda.DeviceBuffer = null,
    d_out: ?cuda.DeviceBuffer = null,
    d_key_cache: ?cuda.DeviceBuffer = null,
    d_val_cache: ?cuda.DeviceBuffer = null,
    gpu_output_norm: ?cuda.DeviceBuffer = null,
    gpu_lm_head: ?cuda.DeviceBuffer = null,

    // Compiled Graph Execution Plan (zgc + PyTorch CUDA execution engine)
    plan: ?graph_mod.ExecutionPlan = null,
    logits_slot_id: ?graph_mod.SlotId = null,

    // Host scratch working buffers
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
            .n_gpu_layers = 0,
            .plan = null,
            .logits_slot_id = null,
            .d_x = null,
            .d_xb = null,
            .d_q = null,
            .d_k = null,
            .d_v = null,
            .d_gate = null,
            .d_up = null,
            .d_out = null,
            .d_key_cache = null,
            .d_val_cache = null,
            .gpu_output_norm = null,
            .gpu_lm_head = null,
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

    fn parseNormWeights(allocator: std.mem.Allocator, norm_data: []const u8, st: format.StorageType, count: usize) ![]f32 {
        const buf = try allocator.alloc(f32, count);
        errdefer allocator.free(buf);

        switch (st) {
            .f32 => {
                for (0..count) |i| {
                    const off = i * 4;
                    if (off + 4 <= norm_data.len) {
                        buf[i] = @bitCast(std.mem.readInt(u32, norm_data[off .. off + 4][0..4], .little));
                    } else {
                        buf[i] = 1.0;
                    }
                }
            },
            .bf16 => {
                for (0..count) |i| {
                    const off = i * 2;
                    if (off + 2 <= norm_data.len) {
                        const raw = std.mem.readInt(u16, norm_data[off .. off + 2][0..2], .little);
                        buf[i] = @bitCast(@as(u32, raw) << 16);
                    } else {
                        buf[i] = 1.0;
                    }
                }
            },
            .f16 => {
                for (0..count) |i| {
                    const off = i * 2;
                    if (off + 2 <= norm_data.len) {
                        const raw = std.mem.readInt(u16, norm_data[off .. off + 2][0..2], .little);
                        buf[i] = @floatCast(@as(f16, @bitCast(raw)));
                    } else {
                        buf[i] = 1.0;
                    }
                }
            },
            else => {
                for (0..count) |i| {
                    const off = i * 4;
                    if (off + 4 <= norm_data.len) {
                        buf[i] = @bitCast(std.mem.readInt(u32, norm_data[off .. off + 4][0..4], .little));
                    } else {
                        buf[i] = 1.0;
                    }
                }
            },
        }
        return buf;
    }

    /// Builds a declarative AOT tensor computation graph (ComputeGraph) for the full transformer model.
    pub fn buildComputeGraph(
        allocator: std.mem.Allocator,
        cfg: ModelConfig,
        tok_embeddings: TensorRef,
        layers: []const LayerWeights,
        output_norm: []const f32,
        lm_head: TensorRef,
        gpu_output_norm: ?cuda.DeviceBuffer,
        gpu_lm_head: ?cuda.DeviceBuffer,
    ) !struct { cg: graph_mod.ComputeGraph, logits_slot: graph_mod.SlotId } {
        var cg = graph_mod.ComputeGraph.init(allocator);
        errdefer cg.deinit();

        const dim = cfg.dim;
        const hidden_dim = cfg.hidden_dim;
        const n_heads = cfg.n_heads;
        const n_kv_heads = cfg.n_kv_heads;
        const head_dim = cfg.head_dim;
        const q_dim = @max(n_heads * head_dim, dim);
        const kv_dim = @max(n_kv_heads * head_dim, dim);

        // Initial activation slot x on host
        var current_slot = try cg.addSlot("x_init", &.{dim}, .f32, .cpu);
        var current_dev: graph_mod.Device = .cpu;

        // Embedding lookup node
        try cg.addNode(.{
            .op = .embedding_lookup,
            .name = "embedding_lookup",
            .inputs = .{ null, null, null },
            .outputs = .{ current_slot, null },
            .weight = .{
                .data = tok_embeddings.data,
                .storage_type = tok_embeddings.storage_type,
                .rows = tok_embeddings.rows,
                .cols = tok_embeddings.cols,
                .gpu_buf = null,
            },
            .dim = dim,
            .device = .cpu,
        });

        for (layers, 0..) |layer, l| {
            const target_dev: graph_mod.Device = if (layer.is_gpu) .{ .cuda = 0 } else .cpu;

            // Transition from CPU to GPU or GPU to CPU if offload boundary changes
            if (!current_dev.eql(target_dev)) {
                const copy_slot = try cg.addSlot(if (target_dev.isCuda()) "x_h2d" else "x_d2h", &.{dim}, .f32, target_dev);
                try cg.addNode(.{
                    .op = if (target_dev.isCuda()) .device_copy_h2d else .device_copy_d2h,
                    .name = if (target_dev.isCuda()) "copy_h2d" else "copy_d2h",
                    .inputs = .{ current_slot, null, null },
                    .outputs = .{ copy_slot, null },
                    .dim = dim,
                    .device = target_dev,
                });
                current_slot = copy_slot;
                current_dev = target_dev;
            }

            // 1. Attention RMSNorm: x -> xb
            const xb_slot = try cg.addSlot("xb_attn", &.{dim}, .f32, current_dev);
            try cg.addNode(.{
                .op = .rmsnorm,
                .name = "attn_norm",
                .inputs = .{ current_slot, null, null },
                .outputs = .{ xb_slot, null },
                .weight = .{
                    .data = std.mem.sliceAsBytes(layer.attn_norm),
                    .storage_type = .f32,
                    .rows = layer.attn_norm.len,
                    .cols = 1,
                    .gpu_buf = layer.gpu_attn_norm,
                },
                .dim = dim,
                .eps = cfg.norm_eps,
                .device = current_dev,
            });

            // 2. Q, K, V Projections
            const q_slot = try cg.addSlot("q", &.{q_dim}, .f32, current_dev);
            try cg.addNode(.{
                .op = .gemv,
                .name = "wq",
                .inputs = .{ xb_slot, null, null },
                .outputs = .{ q_slot, null },
                .weight = .{
                    .data = layer.wq.data,
                    .storage_type = layer.wq.storage_type,
                    .rows = layer.wq.rows,
                    .cols = layer.wq.cols,
                    .gpu_buf = layer.gpu_wq,
                },
                .device = current_dev,
            });

            const k_slot = try cg.addSlot("k", &.{kv_dim}, .f32, current_dev);
            try cg.addNode(.{
                .op = .gemv,
                .name = "wk",
                .inputs = .{ xb_slot, null, null },
                .outputs = .{ k_slot, null },
                .weight = .{
                    .data = layer.wk.data,
                    .storage_type = layer.wk.storage_type,
                    .rows = layer.wk.rows,
                    .cols = layer.wk.cols,
                    .gpu_buf = layer.gpu_wk,
                },
                .device = current_dev,
            });

            const v_slot = try cg.addSlot("v", &.{kv_dim}, .f32, current_dev);
            try cg.addNode(.{
                .op = .gemv,
                .name = "wv",
                .inputs = .{ xb_slot, null, null },
                .outputs = .{ v_slot, null },
                .weight = .{
                    .data = layer.wv.data,
                    .storage_type = layer.wv.storage_type,
                    .rows = layer.wv.rows,
                    .cols = layer.wv.cols,
                    .gpu_buf = layer.gpu_wv,
                },
                .device = current_dev,
            });

            // 3. QK-Norm & RoPE
            if (layer.attn_q_norm != null and layer.attn_k_norm != null and current_dev.isCuda()) {
                try cg.addNode(.{
                    .op = .fused_qknorm_rope,
                    .name = "fused_qknorm_rope",
                    .inputs = .{ q_slot, k_slot, null },
                    .outputs = .{ q_slot, k_slot },
                    .weight = .{
                        .data = std.mem.sliceAsBytes(layer.attn_q_norm.?),
                        .storage_type = .f32,
                        .rows = layer.attn_q_norm.?.len,
                        .cols = 1,
                        .gpu_buf = layer.gpu_attn_q_norm,
                    },
                    .aux_weight = .{
                        .data = std.mem.sliceAsBytes(layer.attn_k_norm.?),
                        .storage_type = .f32,
                        .rows = layer.attn_k_norm.?.len,
                        .cols = 1,
                        .gpu_buf = layer.gpu_attn_k_norm,
                    },
                    .n_heads = n_heads,
                    .n_kv_heads = n_kv_heads,
                    .head_dim = head_dim,
                    .eps = cfg.norm_eps,
                    .rope_theta = cfg.rope_theta,
                    .device = current_dev,
                });
            } else {
                if (layer.attn_q_norm) |qn| {
                    try cg.addNode(.{
                        .op = .head_rmsnorm,
                        .name = "attn_q_norm",
                        .inputs = .{ q_slot, null, null },
                        .outputs = .{ q_slot, null },
                        .weight = .{
                            .data = std.mem.sliceAsBytes(qn),
                            .storage_type = .f32,
                            .rows = qn.len,
                            .cols = 1,
                            .gpu_buf = layer.gpu_attn_q_norm,
                        },
                        .n_heads = n_heads,
                        .head_dim = head_dim,
                        .eps = cfg.norm_eps,
                        .device = current_dev,
                    });
                }
                if (layer.attn_k_norm) |kn| {
                    try cg.addNode(.{
                        .op = .head_rmsnorm,
                        .name = "attn_k_norm",
                        .inputs = .{ k_slot, null, null },
                        .outputs = .{ k_slot, null },
                        .weight = .{
                            .data = std.mem.sliceAsBytes(kn),
                            .storage_type = .f32,
                            .rows = kn.len,
                            .cols = 1,
                            .gpu_buf = layer.gpu_attn_k_norm,
                        },
                        .n_heads = n_kv_heads,
                        .head_dim = head_dim,
                        .eps = cfg.norm_eps,
                        .device = current_dev,
                    });
                }
                try cg.addNode(.{
                    .op = .rope,
                    .name = "rope",
                    .inputs = .{ q_slot, k_slot, null },
                    .outputs = .{ q_slot, k_slot },
                    .n_heads = n_heads,
                    .n_kv_heads = n_kv_heads,
                    .head_dim = head_dim,
                    .rope_theta = cfg.rope_theta,
                    .device = current_dev,
                });
            }

            // 4. Attention GQA
            const attn_out = try cg.addSlot("attn_out", &.{q_dim}, .f32, current_dev);
            try cg.addNode(.{
                .op = .attention_gqa,
                .name = "gqa_attention",
                .inputs = .{ q_slot, k_slot, v_slot },
                .outputs = .{ attn_out, null },
                .layer_idx = l,
                .dim = dim,
                .n_heads = n_heads,
                .n_kv_heads = n_kv_heads,
                .head_dim = head_dim,
                .device = current_dev,
            });

            // 5. Wo Output Projection
            const wo_out = try cg.addSlot("wo_out", &.{dim}, .f32, current_dev);
            try cg.addNode(.{
                .op = .gemv,
                .name = "wo",
                .inputs = .{ attn_out, null, null },
                .outputs = .{ wo_out, null },
                .weight = .{
                    .data = layer.wo.data,
                    .storage_type = layer.wo.storage_type,
                    .rows = layer.wo.rows,
                    .cols = layer.wo.cols,
                    .gpu_buf = layer.gpu_wo,
                },
                .device = current_dev,
            });

            // 6. Residual Connection 1: current_slot = current_slot + wo_out
            try cg.addNode(.{
                .op = .add_residual,
                .name = "residual_attn",
                .inputs = .{ current_slot, wo_out, null },
                .outputs = .{ current_slot, null },
                .dim = dim,
                .device = current_dev,
            });

            // 7. Feed-Forward RMSNorm: current_slot -> xb_ffn
            const xb_ffn = try cg.addSlot("xb_ffn", &.{dim}, .f32, current_dev);
            try cg.addNode(.{
                .op = .rmsnorm,
                .name = "ffn_norm",
                .inputs = .{ current_slot, null, null },
                .outputs = .{ xb_ffn, null },
                .weight = .{
                    .data = std.mem.sliceAsBytes(layer.ffn_norm),
                    .storage_type = .f32,
                    .rows = layer.ffn_norm.len,
                    .cols = 1,
                    .gpu_buf = layer.gpu_ffn_norm,
                },
                .dim = dim,
                .eps = cfg.norm_eps,
                .device = current_dev,
            });

            // 8. Gate & Up projections
            const gate_slot = try cg.addSlot("gate", &.{hidden_dim}, .f32, current_dev);
            try cg.addNode(.{
                .op = .gemv,
                .name = "w_gate",
                .inputs = .{ xb_ffn, null, null },
                .outputs = .{ gate_slot, null },
                .weight = .{
                    .data = layer.w_gate.data,
                    .storage_type = layer.w_gate.storage_type,
                    .rows = layer.w_gate.rows,
                    .cols = layer.w_gate.cols,
                    .gpu_buf = layer.gpu_w_gate,
                },
                .device = current_dev,
            });

            const up_slot = try cg.addSlot("up", &.{hidden_dim}, .f32, current_dev);
            try cg.addNode(.{
                .op = .gemv,
                .name = "w_up",
                .inputs = .{ xb_ffn, null, null },
                .outputs = .{ up_slot, null },
                .weight = .{
                    .data = layer.w_up.data,
                    .storage_type = layer.w_up.storage_type,
                    .rows = layer.w_up.rows,
                    .cols = layer.w_up.cols,
                    .gpu_buf = layer.gpu_w_up,
                },
                .device = current_dev,
            });

            // 9. SwiGLU activation (gate = SiLU(gate) * up)
            try cg.addNode(.{
                .op = .swiglu,
                .name = "swiglu",
                .inputs = .{ gate_slot, up_slot, null },
                .outputs = .{ gate_slot, null },
                .hidden_dim = hidden_dim,
                .device = current_dev,
            });

            // 10. Down projection (w_down * gate -> xb_down)
            const xb_down = try cg.addSlot("xb_down", &.{dim}, .f32, current_dev);
            try cg.addNode(.{
                .op = .gemv,
                .name = "w_down",
                .inputs = .{ gate_slot, null, null },
                .outputs = .{ xb_down, null },
                .weight = .{
                    .data = layer.w_down.data,
                    .storage_type = layer.w_down.storage_type,
                    .rows = layer.w_down.rows,
                    .cols = layer.w_down.cols,
                    .gpu_buf = layer.gpu_w_down,
                },
                .device = current_dev,
            });

            // 11. Residual Connection 2: current_slot = current_slot + xb_down
            try cg.addNode(.{
                .op = .add_residual,
                .name = "residual_ffn",
                .inputs = .{ current_slot, xb_down, null },
                .outputs = .{ current_slot, null },
                .dim = dim,
                .device = current_dev,
            });
        }

        // Final output norm & LM head
        const target_head_dev: graph_mod.Device = if (gpu_lm_head != null and gpu_output_norm != null) .{ .cuda = 0 } else .cpu;
        if (!current_dev.eql(target_head_dev)) {
            const copy_slot = try cg.addSlot(if (target_head_dev.isCuda()) "x_head_h2d" else "x_head_d2h", &.{dim}, .f32, target_head_dev);
            try cg.addNode(.{
                .op = if (target_head_dev.isCuda()) .device_copy_h2d else .device_copy_d2h,
                .name = if (target_head_dev.isCuda()) "copy_head_h2d" else "copy_head_d2h",
                .inputs = .{ current_slot, null, null },
                .outputs = .{ copy_slot, null },
                .dim = dim,
                .device = target_head_dev,
            });
            current_slot = copy_slot;
            current_dev = target_head_dev;
        }

        // Output RMSNorm
        const xb_final = try cg.addSlot("xb_final", &.{dim}, .f32, current_dev);
        try cg.addNode(.{
            .op = .rmsnorm,
            .name = "output_norm",
            .inputs = .{ current_slot, null, null },
            .outputs = .{ xb_final, null },
            .weight = .{
                .data = std.mem.sliceAsBytes(output_norm),
                .storage_type = .f32,
                .rows = output_norm.len,
                .cols = 1,
                .gpu_buf = gpu_output_norm,
            },
            .dim = dim,
            .eps = cfg.norm_eps,
            .device = current_dev,
        });

        // LM Head Projection -> logits
        const logits_dev_slot = try cg.addSlot("logits_raw", &.{cfg.vocab_size}, .f32, current_dev);
        try cg.addNode(.{
            .op = .gemv,
            .name = "lm_head",
            .inputs = .{ xb_final, null, null },
            .outputs = .{ logits_dev_slot, null },
            .weight = .{
                .data = lm_head.data,
                .storage_type = lm_head.storage_type,
                .rows = lm_head.rows,
                .cols = lm_head.cols,
                .gpu_buf = gpu_lm_head,
            },
            .device = current_dev,
        });

        // If logits are on device, copy back to host
        var final_logits_slot = logits_dev_slot;
        if (current_dev.isCuda()) {
            final_logits_slot = try cg.addSlot("logits_host", &.{cfg.vocab_size}, .f32, .cpu);
            try cg.addNode(.{
                .op = .device_copy_d2h,
                .name = "copy_logits_d2h",
                .inputs = .{ logits_dev_slot, null, null },
                .outputs = .{ final_logits_slot, null },
                .dim = cfg.vocab_size,
                .device = .cpu,
            });
        }

        return .{ .cg = cg, .logits_slot = final_logits_slot };
    }

    /// Constructs and initializes a TransformerEngine from an open HKReader
    pub fn initFromReader(allocator: std.mem.Allocator, reader: *reader_mod.HKReader) !*TransformerEngine {
        return initFromReaderWithOptions(allocator, reader, .{});
    }

    /// Constructs and initializes a TransformerEngine with custom offloading options
    pub fn initFromReaderWithOptions(allocator: std.mem.Allocator, reader: *reader_mod.HKReader, options: EngineOptions) !*TransformerEngine {
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
            reader.toc.find("language_model.model.embed_tokens.weight") orelse
            reader.toc.find("mtp.pre_fc_norm_embedding.weight") orelse
            reader.toc.find("mtp.fc.weight") orelse
            return error.MissingEmbeddingTensor;

        cfg.vocab_size = @intCast(emb_entry.shape[0]);
        cfg.dim = if (emb_entry.shape.len > 1 and emb_entry.shape[1] > 0) @intCast(emb_entry.shape[1]) else @intCast(emb_entry.shape[0]);

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
            reader.toc.find("model.layers.0.self_attn.q_proj.weight") orelse
            reader.toc.find("mtp.layers.0.self_attn.q_proj.weight");
        if (wq_0_entry) |wq_e| {
            const wq_rows: usize = @intCast(wq_e.shape[0]);
            if (cfg.head_dim > 0 and wq_rows > 0) {
                cfg.n_heads = @max(1, wq_rows / cfg.head_dim);
            }
        }

        const wk_0_entry = reader.toc.find("blk.0.attn_k.weight") orelse
            reader.toc.find("model.layers.0.self_attn.k_proj.weight") orelse
            reader.toc.find("mtp.layers.0.self_attn.k_proj.weight");
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

        // Find output_norm.weight
        const norm_entry = reader.toc.find("output_norm.weight") orelse
            reader.toc.find("model.norm.weight") orelse
            reader.toc.find("language_model.model.norm.weight") orelse
            reader.toc.find("mtp.norm.weight") orelse
            return error.MissingOutputNormTensor;
        const norm_data = try reader.getTensorData(norm_entry);
        const output_norm_buf = try parseNormWeights(allocator, norm_data, norm_entry.storage_type, cfg.dim);
        errdefer allocator.free(output_norm_buf);

        // Find lm_head / output.weight
        const head_entry_opt = reader.toc.find("output.weight") orelse
            reader.toc.find("lm_head.weight") orelse
            reader.toc.find("mtp.fc.weight");
        const lm_head = if (head_entry_opt) |h_entry| blk: {
            const h_data = try reader.getTensorData(h_entry);
            break :blk TensorRef{
                .data = h_data,
                .storage_type = h_entry.storage_type,
                .rows = @intCast(h_entry.shape[0]),
                .cols = @intCast(h_entry.shape[1]),
            };
        } else tok_embeddings;

        // Detect number of layers
        var layer_count: usize = 0;
        var name_buf: [64]u8 = undefined;
        while (true) : (layer_count += 1) {
            const test_name = std.fmt.bufPrint(&name_buf, "blk.{}.attn_q.weight", .{layer_count}) catch break;
            if (reader.toc.find(test_name) == null) {
                const alt_name = std.fmt.bufPrint(&name_buf, "model.layers.{}.self_attn.q_proj.weight", .{layer_count}) catch break;
                if (reader.toc.find(alt_name) == null) {
                    const mtp_name = std.fmt.bufPrint(&name_buf, "mtp.layers.{}.self_attn.q_proj.weight", .{layer_count}) catch break;
                    if (reader.toc.find(mtp_name) == null) break;
                }
            }
        }
        if (layer_count == 0) return error.NoLayersFound;
        cfg.n_layers = layer_count;

        // Intermediate dimension from blk.0.ffn_gate.weight
        const gate_0_entry = reader.toc.find("blk.0.ffn_gate.weight") orelse
            reader.toc.find("model.layers.0.mlp.gate_proj.weight") orelse
            reader.toc.find("mtp.layers.0.mlp.gate_proj.weight") orelse
            return error.MissingFFNGateTensor;
        cfg.hidden_dim = @intCast(gate_0_entry.shape[0]);

        const engine = try allocator.create(TransformerEngine);
        errdefer allocator.destroy(engine);

        engine.* = try TransformerEngine.init(allocator, cfg, tok_embeddings, output_norm_buf, lm_head);
        engine.owns_output_norm = true;
        errdefer engine.deinit();

        // Determine number of GPU layers to offload:
        // By default, offload as high as possible (all layers), dynamically
        // allocating VRAM for each layer until VRAM budget is reached or all layers offloaded.
        const target_gpu_layers = options.gpu_layers orelse cfg.n_layers;
        const n_gpu = if (cuda.isAvailable()) @min(target_gpu_layers, cfg.n_layers) else 0;

        var actual_gpu_layers: usize = 0;
        var can_offload_more: bool = (n_gpu > 0 and cuda.isAvailable());

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

            const attn_norm_e = reader.toc.find(attn_norm_name) orelse
                reader.toc.find(std.fmt.bufPrint(&buf1, "model.layers.{}.input_layernorm.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf1, "layers.{}.input_layernorm.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf1, "mtp.layers.{}.input_layernorm.weight", .{l}) catch "") orelse
                return error.MissingLayerTensor;
            const wq_e = reader.toc.find(wq_name) orelse
                reader.toc.find(std.fmt.bufPrint(&buf2, "model.layers.{}.self_attn.q_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf2, "layers.{}.q_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf2, "mtp.layers.{}.self_attn.q_proj.weight", .{l}) catch "") orelse
                return error.MissingLayerTensor;
            const wk_e = reader.toc.find(wk_name) orelse
                reader.toc.find(std.fmt.bufPrint(&buf3, "model.layers.{}.self_attn.k_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf3, "layers.{}.k_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf3, "mtp.layers.{}.self_attn.k_proj.weight", .{l}) catch "") orelse
                return error.MissingLayerTensor;
            const wv_e = reader.toc.find(wv_name) orelse
                reader.toc.find(std.fmt.bufPrint(&buf4, "model.layers.{}.self_attn.v_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf4, "layers.{}.v_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf4, "mtp.layers.{}.self_attn.v_proj.weight", .{l}) catch "") orelse
                return error.MissingLayerTensor;
            const wo_e = reader.toc.find(wo_name) orelse
                reader.toc.find(std.fmt.bufPrint(&buf5, "model.layers.{}.self_attn.o_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf5, "layers.{}.out_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf5, "mtp.layers.{}.self_attn.o_proj.weight", .{l}) catch "") orelse
                return error.MissingLayerTensor;
            const ffn_norm_e = reader.toc.find(ffn_norm_name) orelse
                reader.toc.find(std.fmt.bufPrint(&buf6, "model.layers.{}.post_attention_layernorm.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf6, "layers.{}.post_attention_layernorm.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf6, "mtp.layers.{}.post_attention_layernorm.weight", .{l}) catch "") orelse
                return error.MissingLayerTensor;
            const wgate_e = reader.toc.find(wgate_name) orelse
                reader.toc.find(std.fmt.bufPrint(&buf7, "model.layers.{}.mlp.gate_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf7, "layers.{}.gate_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf7, "layers.{}.mlp_fc1.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf7, "mtp.layers.{}.mlp.gate_proj.weight", .{l}) catch "") orelse
                return error.MissingLayerTensor;
            const wup_e = reader.toc.find(wup_name) orelse
                reader.toc.find(std.fmt.bufPrint(&buf8, "model.layers.{}.mlp.up_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf8, "layers.{}.up_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf8, "layers.{}.mlp_fc1.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf8, "mtp.layers.{}.mlp.up_proj.weight", .{l}) catch "") orelse
                return error.MissingLayerTensor;
            const wdown_e = reader.toc.find(wdown_name) orelse
                reader.toc.find(std.fmt.bufPrint(&buf9, "model.layers.{}.mlp.down_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf9, "layers.{}.down_proj.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf9, "layers.{}.mlp_fc2.weight", .{l}) catch "") orelse
                reader.toc.find(std.fmt.bufPrint(&buf9, "mtp.layers.{}.mlp.down_proj.weight", .{l}) catch "") orelse
                return error.MissingLayerTensor;

            const attn_norm_data = try reader.getTensorData(attn_norm_e);
            const ffn_norm_data = try reader.getTensorData(ffn_norm_e);

            const a_norm_buf = try parseNormWeights(allocator, attn_norm_data, attn_norm_e.storage_type, cfg.dim);
            errdefer allocator.free(a_norm_buf);

            const f_norm_buf = try parseNormWeights(allocator, ffn_norm_data, ffn_norm_e.storage_type, cfg.dim);
            errdefer allocator.free(f_norm_buf);

            // QK-Norm Detection (Qwen3 / Gemma 2)
            var buf_qn: [64]u8 = undefined;
            var buf_kn: [64]u8 = undefined;
            const qn_name = std.fmt.bufPrint(&buf_qn, "blk.{}.attn_q_norm.weight", .{l}) catch "";
            const kn_name = std.fmt.bufPrint(&buf_kn, "blk.{}.attn_k_norm.weight", .{l}) catch "";
            const qn_e = reader.toc.find(qn_name) orelse reader.toc.find(std.fmt.bufPrint(&buf_qn, "model.layers.{}.self_attn.q_norm.weight", .{l}) catch "");
            const kn_e = reader.toc.find(kn_name) orelse reader.toc.find(std.fmt.bufPrint(&buf_kn, "model.layers.{}.self_attn.k_norm.weight", .{l}) catch "");

            var qn_buf: ?[]f32 = null;
            if (qn_e) |entry| {
                const data = try reader.getTensorData(entry);
                const count: usize = if (entry.shape[0] > 0) @intCast(entry.shape[0]) else cfg.head_dim;
                qn_buf = try parseNormWeights(allocator, data, entry.storage_type, count);
            }

            var kn_buf: ?[]f32 = null;
            if (kn_e) |entry| {
                const data = try reader.getTensorData(entry);
                const count: usize = if (entry.shape[0] > 0) @intCast(entry.shape[0]) else cfg.head_dim;
                kn_buf = try parseNormWeights(allocator, data, entry.storage_type, count);
            }

            const wq_data = try reader.getTensorData(wq_e);
            const wk_data = try reader.getTensorData(wk_e);
            const wv_data = try reader.getTensorData(wv_e);
            const wo_data = try reader.getTensorData(wo_e);
            const wgate_data = try reader.getTensorData(wgate_e);
            const wup_data = try reader.getTensorData(wup_e);
            const wdown_data = try reader.getTensorData(wdown_e);

            var gpu_wq: ?cuda.DeviceBuffer = null;
            var gpu_wk: ?cuda.DeviceBuffer = null;
            var gpu_wv: ?cuda.DeviceBuffer = null;
            var gpu_wo: ?cuda.DeviceBuffer = null;
            var gpu_w_gate: ?cuda.DeviceBuffer = null;
            var gpu_w_up: ?cuda.DeviceBuffer = null;
            var gpu_w_down: ?cuda.DeviceBuffer = null;
            var gpu_a_norm: ?cuda.DeviceBuffer = null;
            var gpu_f_norm: ?cuda.DeviceBuffer = null;
            var gpu_qn: ?cuda.DeviceBuffer = null;
            var gpu_kn: ?cuda.DeviceBuffer = null;

            var is_gpu_layer = false;
            if (l < n_gpu and can_offload_more) {
                gpu_wq = cuda.DeviceBuffer.upload(wq_data) catch null;
                gpu_wk = cuda.DeviceBuffer.upload(wk_data) catch null;
                gpu_wv = cuda.DeviceBuffer.upload(wv_data) catch null;
                gpu_wo = cuda.DeviceBuffer.upload(wo_data) catch null;
                gpu_w_gate = cuda.DeviceBuffer.upload(wgate_data) catch null;
                gpu_w_up = cuda.DeviceBuffer.upload(wup_data) catch null;
                gpu_w_down = cuda.DeviceBuffer.upload(wdown_data) catch null;
                gpu_a_norm = cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(a_norm_buf)) catch null;
                gpu_f_norm = cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(f_norm_buf)) catch null;
                if (qn_buf) |qn| gpu_qn = cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(qn)) catch null;
                if (kn_buf) |kn| gpu_kn = cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(kn)) catch null;

                const all_uploaded = (gpu_wq != null and gpu_wk != null and gpu_wv != null and gpu_wo != null and
                    gpu_w_gate != null and gpu_w_up != null and gpu_w_down != null and
                    gpu_a_norm != null and gpu_f_norm != null and
                    (qn_buf == null or gpu_qn != null) and (kn_buf == null or gpu_kn != null));

                if (all_uploaded) {
                    is_gpu_layer = true;
                    actual_gpu_layers += 1;
                } else {
                    // Out of VRAM or allocation error: clean up partial buffers and keep remaining layers on CPU
                    if (gpu_wq) |b| b.free(); gpu_wq = null;
                    if (gpu_wk) |b| b.free(); gpu_wk = null;
                    if (gpu_wv) |b| b.free(); gpu_wv = null;
                    if (gpu_wo) |b| b.free(); gpu_wo = null;
                    if (gpu_w_gate) |b| b.free(); gpu_w_gate = null;
                    if (gpu_w_up) |b| b.free(); gpu_w_up = null;
                    if (gpu_w_down) |b| b.free(); gpu_w_down = null;
                    if (gpu_a_norm) |b| b.free(); gpu_a_norm = null;
                    if (gpu_f_norm) |b| b.free(); gpu_f_norm = null;
                    if (gpu_qn) |b| b.free(); gpu_qn = null;
                    if (gpu_kn) |b| b.free(); gpu_kn = null;
                    can_offload_more = false;
                }
            }

            try engine.layers.append(allocator, .{
                .attn_norm = a_norm_buf,
                .wq = .{ .data = wq_data, .storage_type = wq_e.storage_type, .rows = @intCast(wq_e.shape[0]), .cols = @intCast(wq_e.shape[1]) },
                .wk = .{ .data = wk_data, .storage_type = wk_e.storage_type, .rows = @intCast(wk_e.shape[0]), .cols = @intCast(wk_e.shape[1]) },
                .wv = .{ .data = wv_data, .storage_type = wv_e.storage_type, .rows = @intCast(wv_e.shape[0]), .cols = @intCast(wv_e.shape[1]) },
                .wo = .{ .data = wo_data, .storage_type = wo_e.storage_type, .rows = @intCast(wo_e.shape[0]), .cols = @intCast(wo_e.shape[1]) },
                .ffn_norm = f_norm_buf,
                .w_gate = .{ .data = wgate_data, .storage_type = wgate_e.storage_type, .rows = @intCast(wgate_e.shape[0]), .cols = @intCast(wgate_e.shape[1]) },
                .w_up = .{ .data = wup_data, .storage_type = wup_e.storage_type, .rows = @intCast(wup_e.shape[0]), .cols = @intCast(wup_e.shape[1]) },
                .w_down = .{ .data = wdown_data, .storage_type = wdown_e.storage_type, .rows = @intCast(wdown_e.shape[0]), .cols = @intCast(wdown_e.shape[1]) },
                .owns_norms = true,
                .attn_q_norm = qn_buf,
                .attn_k_norm = kn_buf,
                .gpu_wq = gpu_wq,
                .gpu_wk = gpu_wk,
                .gpu_wv = gpu_wv,
                .gpu_wo = gpu_wo,
                .gpu_w_gate = gpu_w_gate,
                .gpu_w_up = gpu_w_up,
                .gpu_w_down = gpu_w_down,
                .gpu_attn_norm = gpu_a_norm,
                .gpu_ffn_norm = gpu_f_norm,
                .gpu_attn_q_norm = gpu_qn,
                .gpu_attn_k_norm = gpu_kn,
                .is_gpu = is_gpu_layer,
            });
        }

        engine.n_gpu_layers = actual_gpu_layers;

        // Allocate device scratch buffers if at least one layer is offloaded
        if (actual_gpu_layers > 0 and cuda.isAvailable()) {
            const kv_dim = @max(cfg.n_kv_heads * cfg.head_dim, cfg.dim);
            const max_q_dim = @max(cfg.n_heads * cfg.head_dim, cfg.dim);

            engine.d_x = cuda.DeviceBuffer.allocUninit(cfg.dim * @sizeOf(f32)) catch null;
            engine.d_xb = cuda.DeviceBuffer.allocUninit(cfg.dim * @sizeOf(f32)) catch null;
            engine.d_q = cuda.DeviceBuffer.allocUninit(max_q_dim * @sizeOf(f32)) catch null;
            engine.d_k = cuda.DeviceBuffer.allocUninit(kv_dim * @sizeOf(f32)) catch null;
            engine.d_v = cuda.DeviceBuffer.allocUninit(kv_dim * @sizeOf(f32)) catch null;
            engine.d_gate = cuda.DeviceBuffer.allocUninit(cfg.hidden_dim * @sizeOf(f32)) catch null;
            engine.d_up = cuda.DeviceBuffer.allocUninit(cfg.hidden_dim * @sizeOf(f32)) catch null;
            engine.d_out = cuda.DeviceBuffer.allocUninit(@max(cfg.vocab_size, cfg.dim) * @sizeOf(f32)) catch null;
            engine.d_key_cache = cuda.DeviceBuffer.allocUninit(cfg.n_layers * cfg.max_seq_len * kv_dim * @sizeOf(f32)) catch null;
            engine.d_val_cache = cuda.DeviceBuffer.allocUninit(cfg.n_layers * cfg.max_seq_len * kv_dim * @sizeOf(f32)) catch null;

            if (actual_gpu_layers == cfg.n_layers) {
                engine.gpu_output_norm = cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(output_norm_buf)) catch null;
                engine.gpu_lm_head = cuda.DeviceBuffer.upload(lm_head.data) catch null;
            }
        }

        // Ahead-of-Time Computation Graph compilation (zgc + PyTorch CUDA execution engine)
        var graph_res = buildComputeGraph(
            allocator,
            cfg,
            tok_embeddings,
            engine.layers.items,
            output_norm_buf,
            lm_head,
            engine.gpu_output_norm,
            engine.gpu_lm_head,
        ) catch null;

        if (graph_res) |*gres| {
            defer gres.cg.deinit();
            engine.plan = graph_mod.ExecutionPlan.compile(allocator, &gres.cg) catch null;
            engine.logits_slot_id = gres.logits_slot;
        }

        return engine;
    }

    pub fn deinit(self: *TransformerEngine) void {
        if (self.plan) |*p| {
            p.deinit();
            self.plan = null;
        }

        for (self.layers.items) |layer| {
            if (layer.owns_norms) {
                self.allocator.free(layer.attn_norm);
                self.allocator.free(layer.ffn_norm);
                if (layer.attn_q_norm) |qn| self.allocator.free(qn);
                if (layer.attn_k_norm) |kn| self.allocator.free(kn);
            }
            if (layer.is_gpu) {
                if (layer.gpu_wq) |b| b.free();
                if (layer.gpu_wk) |b| b.free();
                if (layer.gpu_wv) |b| b.free();
                if (layer.gpu_wo) |b| b.free();
                if (layer.gpu_w_gate) |b| b.free();
                if (layer.gpu_w_up) |b| b.free();
                if (layer.gpu_w_down) |b| b.free();
                if (layer.gpu_attn_norm) |b| b.free();
                if (layer.gpu_ffn_norm) |b| b.free();
                if (layer.gpu_attn_q_norm) |b| b.free();
                if (layer.gpu_attn_k_norm) |b| b.free();
            }
        }
        self.layers.deinit(self.allocator);
        self.cache.deinit();
        if (self.owns_output_norm) {
            self.allocator.free(self.output_norm);
        }

        if (self.d_x) |b| b.free();
        if (self.d_xb) |b| b.free();
        if (self.d_q) |b| b.free();
        if (self.d_k) |b| b.free();
        if (self.d_v) |b| b.free();
        if (self.d_gate) |b| b.free();
        if (self.d_up) |b| b.free();
        if (self.d_out) |b| b.free();
        if (self.d_key_cache) |b| b.free();
        if (self.d_val_cache) |b| b.free();
        if (self.gpu_output_norm) |b| b.free();
        if (self.gpu_lm_head) |b| b.free();

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
            .bf16 => {
                if (std.mem.isAligned(@intFromPtr(weight.data.ptr), @alignOf(u16))) {
                    const w_u16: [*]const u16 = @ptrCast(@alignCast(weight.data.ptr));
                    const total = @min(weight.rows * weight.cols, weight.data.len / 2);
                    tensor_ops.gemvBF16(w_u16[0..total], in_x, bias, out_y[0..safe_rows], safe_rows, weight.cols);
                } else {
                    for (0..safe_rows) |r| {
                        var dot: f32 = 0.0;
                        const row_offset = r * weight.cols * 2;
                        for (0..weight.cols) |c| {
                            const off = row_offset + c * 2;
                            if (off + 2 <= weight.data.len) {
                                const u_val = std.mem.readInt(u16, weight.data[off .. off + 2][0..2], .little);
                                dot += tensor_ops.bf16ToF32(u_val) * in_x[c];
                            }
                        }
                        if (bias) |b| {
                            if (r < b.len) dot += b[r];
                        }
                        out_y[r] = dot;
                    }
                }
            },
            .f16 => {
                for (0..safe_rows) |r| {
                    var dot: f32 = 0.0;
                    const row_offset = r * weight.cols * 2;
                    for (0..weight.cols) |c| {
                        const off = row_offset + c * 2;
                        if (off + 2 <= weight.data.len) {
                            const u_val = std.mem.readInt(u16, weight.data[off .. off + 2][0..2], .little);
                            const f_val: f32 = @floatCast(@as(f16, @bitCast(u_val)));
                            dot += f_val * in_x[c];
                        }
                    }
                    if (bias) |b| {
                        if (r < b.len) dot += b[r];
                    }
                    out_y[r] = dot;
                }
            },
            else => {
                for (0..safe_rows) |r| {
                    out_y[r] = if (bias) |b| (if (r < b.len) b[r] else 0.0) else 0.0;
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
            .bf16 => {
                for (0..dim) |i| {
                    const off = (tid * dim + i) * 2;
                    if (off + 2 <= self.tok_embeddings.data.len) {
                        const u_val = std.mem.readInt(u16, self.tok_embeddings.data[off .. off + 2][0..2], .little);
                        out_x[i] = tensor_ops.bf16ToF32(u_val);
                    } else {
                        out_x[i] = 0.0;
                    }
                }
            },
            .f16 => {
                for (0..dim) |i| {
                    const off = (tid * dim + i) * 2;
                    if (off + 2 <= self.tok_embeddings.data.len) {
                        const u_val = std.mem.readInt(u16, self.tok_embeddings.data[off .. off + 2][0..2], .little);
                        out_x[i] = @floatCast(@as(f16, @bitCast(u_val)));
                    } else {
                        out_x[i] = 0.0;
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
    /// Dispatches through the compiled AOT execution plan (ComputeGraph + MemoryPlanner + CudaGraph),
    /// guaranteeing zero runtime heap allocations and optimal hardware throughput.
    pub fn forward(self: *TransformerEngine, token: u32, pos: usize) []const f32 {
        if (self.plan) |*p| {
            var ctx = graph_mod.StepContext{
                .arena = &p.arena,
                .pos = pos,
                .token_id = token,
                .key_cache = self.cache.key_cache,
                .val_cache = self.cache.val_cache,
                .d_key_cache = if (self.d_key_cache) |b| b.ptr else null,
                .d_val_cache = if (self.d_val_cache) |b| b.ptr else null,
                .max_seq_len = self.config.max_seq_len,
                .att = self.att,
            };
            p.execute(&ctx) catch {
                return self.forwardLegacy(token, pos);
            };
            const logits_slot = &p.slots[self.logits_slot_id.?];
            return p.arena.hostSlice(logits_slot);
        }
        return self.forwardLegacy(token, pos);
    }

    /// Procedural reference execution path (used as robust fallback if graph plan is disabled)
    pub fn forwardLegacy(self: *TransformerEngine, token: u32, pos: usize) []const f32 {
        const cfg = self.config;
        const dim = cfg.dim;
        const head_dim = cfg.head_dim;
        const n_heads = cfg.n_heads;
        const n_kv_heads = cfg.n_kv_heads;
        const kv_dim = @max(n_kv_heads * head_dim, dim);
        const n_rep = if (n_kv_heads > 0) @max(1, n_heads / n_kv_heads) else 1;

        // 1. Embedding lookup on host
        self.lookupEmbedding(token, self.x);

        var x_on_device = false;

        // 2. Transformer layers
        for (self.layers.items, 0..) |layer, l| {
            if (layer.is_gpu and self.d_x != null) {
                if (!x_on_device) {
                    self.d_x.?.copyFromHost(std.mem.sliceAsBytes(self.x)) catch {};
                    x_on_device = true;
                }

                // Attention RMSNorm on device
                cuda.rmsNorm(self.d_x.?, layer.gpu_attn_norm.?, self.d_xb.?, dim, cfg.norm_eps) catch {};

                // Compute Q, K, V projections on device
                cuda.matVec(layer.gpu_wq.?, layer.wq.storage_type, self.d_xb.?, self.d_q.?, layer.wq.rows, layer.wq.cols) catch {};
                cuda.matVec(layer.gpu_wk.?, layer.wk.storage_type, self.d_xb.?, self.d_k.?, layer.wk.rows, layer.wk.cols) catch {};
                cuda.matVec(layer.gpu_wv.?, layer.wv.storage_type, self.d_xb.?, self.d_v.?, layer.wv.rows, layer.wv.cols) catch {};

                // QK-Norm on device if present (Qwen3 / Gemma 2)
                if (layer.gpu_attn_q_norm) |gpu_qn| {
                    const qn_len = if (layer.attn_q_norm) |qn| qn.len else head_dim;
                    cuda.headRmsNorm(self.d_q.?, gpu_qn, n_heads, head_dim, qn_len, cfg.norm_eps) catch {};
                }
                if (layer.gpu_attn_k_norm) |gpu_kn| {
                    const kn_len = if (layer.attn_k_norm) |kn| kn.len else head_dim;
                    cuda.headRmsNorm(self.d_k.?, gpu_kn, n_kv_heads, head_dim, kn_len, cfg.norm_eps) catch {};
                }

                // RoPE on device
                if (head_dim >= 2) {
                    cuda.rope(self.d_q.?, self.d_k.?, pos, n_heads, n_kv_heads, head_dim, cfg.rope_theta) catch {};
                }

                // KV Cache update on device
                cuda.kvCacheUpdate(self.d_key_cache.?, self.d_val_cache.?, self.d_k.?, self.d_v.?, l, pos, cfg.max_seq_len, kv_dim) catch {};

                // GQA Attention on device
                cuda.gqaAttention(self.d_q.?, self.d_key_cache.?, self.d_val_cache.?, self.d_xb.?, l, pos, n_heads, n_kv_heads, head_dim, cfg.max_seq_len, kv_dim) catch {};

                // Wo output projection on device: wo * xb -> q
                cuda.matVec(layer.gpu_wo.?, layer.wo.storage_type, self.d_xb.?, self.d_q.?, layer.wo.rows, layer.wo.cols) catch {};

                // Residual connection on device: x = x + q
                cuda.addResidual(self.d_x.?, self.d_q.?, dim) catch {};

                // Feed-Forward RMSNorm on device
                cuda.rmsNorm(self.d_x.?, layer.gpu_ffn_norm.?, self.d_xb.?, dim, cfg.norm_eps) catch {};

                // Gate & Up projections on device
                cuda.matVec(layer.gpu_w_gate.?, layer.w_gate.storage_type, self.d_xb.?, self.d_gate.?, layer.w_gate.rows, layer.w_gate.cols) catch {};
                cuda.matVec(layer.gpu_w_up.?, layer.w_up.storage_type, self.d_xb.?, self.d_up.?, layer.w_up.rows, layer.w_up.cols) catch {};

                // SwiGLU activation on device: gate = SiLU(gate) * up
                cuda.swiglu(self.d_gate.?, self.d_up.?, cfg.hidden_dim) catch {};

                // Down projection on device: w_down * gate -> xb
                cuda.matVec(layer.gpu_w_down.?, layer.w_down.storage_type, self.d_gate.?, self.d_xb.?, layer.w_down.rows, layer.w_down.cols) catch {};

                // Residual connection on device: x = x + xb
                cuda.addResidual(self.d_x.?, self.d_xb.?, dim) catch {};

                // Determine if transition to host is needed
                const next_is_cpu = if (l + 1 < self.layers.items.len) !self.layers.items[l + 1].is_gpu else (self.gpu_lm_head == null);
                if (next_is_cpu) {
                    self.d_x.?.download(std.mem.sliceAsBytes(self.x)) catch {};
                    x_on_device = false;
                }
            } else {
                // CPU Layer execution
                if (x_on_device and self.d_x != null) {
                    self.d_x.?.download(std.mem.sliceAsBytes(self.x)) catch {};
                    x_on_device = false;
                }

                // Attention RMSNorm
                tensor_ops.rmsNormF32(self.x, layer.attn_norm, cfg.norm_eps, self.xb);

                // Compute Q, K, V projections
                matVec(layer.wq, self.xb, null, self.q);
                matVec(layer.wk, self.xb, null, self.k);
                matVec(layer.wv, self.xb, null, self.v);

                // QK-Norm before RoPE (Qwen3 / Gemma 2)
                if (layer.attn_q_norm) |q_norm| {
                    if (q_norm.len == head_dim) {
                        for (0..n_heads) |h| {
                            const q_head = self.q[h * head_dim .. (h + 1) * head_dim];
                            tensor_ops.rmsNormF32(q_head, q_norm, cfg.norm_eps, q_head);
                        }
                    } else {
                        const norm_len = @min(q_norm.len, self.q.len);
                        for (0..n_heads) |h| {
                            if ((h + 1) * head_dim <= norm_len) {
                                const q_head = self.q[h * head_dim .. (h + 1) * head_dim];
                                const q_w = q_norm[h * head_dim .. (h + 1) * head_dim];
                                tensor_ops.rmsNormF32(q_head, q_w, cfg.norm_eps, q_head);
                            }
                        }
                    }
                }

                if (layer.attn_k_norm) |k_norm| {
                    if (k_norm.len == head_dim) {
                        for (0..n_kv_heads) |h| {
                            const k_head = self.k[h * head_dim .. (h + 1) * head_dim];
                            tensor_ops.rmsNormF32(k_head, k_norm, cfg.norm_eps, k_head);
                        }
                    } else {
                        const norm_len = @min(k_norm.len, self.k.len);
                        for (0..n_kv_heads) |h| {
                            if ((h + 1) * head_dim <= norm_len) {
                                const k_head = self.k[h * head_dim .. (h + 1) * head_dim];
                                const k_w = k_norm[h * head_dim .. (h + 1) * head_dim];
                                tensor_ops.rmsNormF32(k_head, k_w, cfg.norm_eps, k_head);
                            }
                        }
                    }
                }

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
                        if ((kv_head_idx + 1) * head_dim > k_t.len) continue;
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
                        if ((kv_head_idx + 1) * head_dim > v_t.len) continue;
                        const v_head = v_t[kv_head_idx * head_dim .. (kv_head_idx + 1) * head_dim];
                        for (0..head_dim) |d| {
                            out_head[d] += a * v_head[d];
                        }
                    }
                }

                // Attention Output Projection: wo * xb -> q
                matVec(layer.wo, self.xb, null, self.q);

                // Residual connection: x = x + q
                const dim_len = @min(@min(dim, self.x.len), self.q.len);
                for (0..dim_len) |i| {
                    self.x[i] += self.q[i];
                }

                // Feed-Forward SwiGLU Block
                tensor_ops.rmsNormF32(self.x, layer.ffn_norm, cfg.norm_eps, self.xb);
                matVec(layer.w_gate, self.xb, null, self.gate);
                matVec(layer.w_up, self.xb, null, self.up);

                // SiLU(gate) * up
                const hidden_len = @min(@min(cfg.hidden_dim, self.gate.len), self.up.len);
                for (0..hidden_len) |i| {
                    const g = self.gate[i];
                    const silu_g = g / (1.0 + @exp(-g));
                    self.gate[i] = silu_g * self.up[i];
                }

                // Down projection: w_down * gate -> xb
                matVec(layer.w_down, self.gate, null, self.xb);

                // Residual connection: x = x + xb
                const xb_len = @min(@min(dim, self.x.len), self.xb.len);
                for (0..xb_len) |i| {
                    self.x[i] += self.xb[i];
                }
            }
        }

        // 3. Output RMSNorm & LM Head projection
        if (x_on_device and self.gpu_lm_head != null and self.gpu_output_norm != null and self.d_x != null and self.d_xb != null and self.d_out != null) {
            cuda.rmsNorm(self.d_x.?, self.gpu_output_norm.?, self.d_xb.?, dim, cfg.norm_eps) catch {};
            cuda.matVec(self.gpu_lm_head.?, self.lm_head.storage_type, self.d_xb.?, self.d_out.?, self.lm_head.rows, self.lm_head.cols) catch {};
            cuda.synchronize() catch {};
            self.d_out.?.download(std.mem.sliceAsBytes(self.logits)) catch {};
        } else {
            if (x_on_device and self.d_x != null) {
                self.d_x.?.download(std.mem.sliceAsBytes(self.x)) catch {};
            }
            tensor_ops.rmsNormF32(self.x, self.output_norm, cfg.norm_eps, self.xb);
            matVec(self.lm_head, self.xb, null, self.logits);
        }

        return self.logits;
    }
};
