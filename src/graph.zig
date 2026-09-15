//! HK Tensor Graph Compiler (`src/graph.zig`)
//!
//! Inspired by ahead-of-time graph compilation principles (e.g. zgc / Zig Graph Compiler)
//! and unified device execution frameworks.
//!
//! Features:
//! 1. Declarative Computation Graph (DAG) with typed tensor slots and operation nodes.
//! 2. Ahead-of-Time Memory Planner: Lifetime interval analysis (birth/death steps)
//!    with 128-byte aligned memory reuse, pre-allocating a single static
//!    contiguous activation arena with zero runtime heap allocations.
//! 3. Automatic Device Scheduling: Dispatches nodes to Host (CPU SIMD) or Device (CUDA)
//!    and transparently manages boundary synchronization.
//! 4. Hardware Graph Capture: Supports recording the execution plan into a CUDA Graph
//!    for single-launch replay with zero CPU dispatch latency.

const std = @import("std");
const format = @import("format.zig");
const tensor_ops = @import("tensor_ops.zig");
const cuda = @import("cuda.zig");

pub const SlotId = u32;

pub const Device = union(enum) {
    cpu,
    cuda: i32,

    pub fn isCuda(self: Device) bool {
        return self == .cuda;
    }

    pub fn isCpu(self: Device) bool {
        return self == .cpu;
    }

    pub fn eql(self: Device, other: Device) bool {
        return switch (self) {
            .cpu => other == .cpu,
            .cuda => |id1| switch (other) {
                .cuda => |id2| id1 == id2,
                else => false,
            },
        };
    }
};

pub const OpType = enum(u16) {
    embedding_lookup,
    rmsnorm,
    head_rmsnorm, // QK-Norm
    rope,
    gemv,
    attention_gqa,
    swiglu,
    add_residual,
    device_copy_h2d,
    device_copy_d2h,
    fused_qknorm_rope,
    fused_swiglu_residual,
};

pub const TensorSlot = struct {
    id: SlotId,
    name: []const u8,
    shape: [4]usize = .{ 0, 0, 0, 0 },
    ndim: u8 = 1,
    storage_type: format.StorageType = .f32,
    device: Device = .cpu,
    byte_size: usize = 0,
    arena_offset: usize = 0,
    birth_step: usize = 0,
    death_step: usize = 0,

    pub fn numel(self: *const TensorSlot) usize {
        var n: usize = 1;
        for (0..self.ndim) |i| {
            n *= self.shape[i];
        }
        return n;
    }
};

pub const WeightRef = struct {
    data: []const u8,
    storage_type: format.StorageType = .f32,
    rows: usize = 0,
    cols: usize = 0,
    gpu_buf: ?cuda.DeviceBuffer = null,
};

pub const Node = struct {
    op: OpType,
    name: []const u8,
    inputs: [3]?SlotId = .{ null, null, null },
    outputs: [2]?SlotId = .{ null, null },
    weight: ?WeightRef = null,
    aux_weight: ?WeightRef = null,
    device: Device = .cpu,

    // Hyperparameters
    dim: usize = 0,
    hidden_dim: usize = 0,
    n_heads: usize = 0,
    n_kv_heads: usize = 0,
    head_dim: usize = 0,
    layer_idx: usize = 0,
    eps: f32 = 1e-6,
    rope_theta: f32 = 10000.0,
};

/// Pre-allocated contiguous activation memory arena.
pub const ActivationArena = struct {
    allocator: std.mem.Allocator,
    host_bytes: []align(128) u8,
    device_buf: ?cuda.DeviceBuffer = null,
    host_size: usize,
    device_size: usize,

    pub fn init(allocator: std.mem.Allocator, host_size: usize, device_size: usize) !ActivationArena {
        const aligned_host_size = (host_size + 127) & ~@as(usize, 127);
        const host_mem = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(128), @max(aligned_host_size, 128));
        @memset(host_mem, 0);

        var dev_buf: ?cuda.DeviceBuffer = null;
        if (device_size > 0 and cuda.isAvailable()) {
            dev_buf = cuda.DeviceBuffer.allocUninit(device_size) catch null;
        }

        return ActivationArena{
            .allocator = allocator,
            .host_bytes = host_mem,
            .device_buf = dev_buf,
            .host_size = aligned_host_size,
            .device_size = device_size,
        };
    }

    pub fn deinit(self: *ActivationArena) void {
        self.allocator.free(self.host_bytes);
        if (self.device_buf) |b| {
            b.free();
            self.device_buf = null;
        }
    }

    pub inline fn hostSlice(self: *ActivationArena, slot: *const TensorSlot) []f32 {
        const ptr: [*]align(128) u8 = @alignCast(self.host_bytes.ptr + slot.arena_offset);
        const f32_ptr: [*]f32 = @ptrCast(@alignCast(ptr));
        return f32_ptr[0 .. slot.byte_size / @sizeOf(f32)];
    }

    pub inline fn hostSliceConst(self: *const ActivationArena, slot: *const TensorSlot) []const f32 {
        const ptr: [*]align(128) const u8 = @alignCast(self.host_bytes.ptr + slot.arena_offset);
        const f32_ptr: [*]const f32 = @ptrCast(@alignCast(ptr));
        return f32_ptr[0 .. slot.byte_size / @sizeOf(f32)];
    }

    pub inline fn devicePtr(self: *const ActivationArena, slot: *const TensorSlot) ?*anyopaque {
        if (self.device_buf) |b| {
            const byte_ptr: [*]u8 = @ptrCast(b.ptr);
            return @ptrCast(byte_ptr + slot.arena_offset);
        }
        return null;
    }
};

/// Ahead-Of-Time Memory Planner.
/// Computes lifetime intervals (birth and death step) for all intermediate activations
/// and assigns 128-byte aligned offsets into a unified scratch arena, reusing memory
/// for disjoint lifetimes.
fn planDevice(
    allocator: std.mem.Allocator,
    slots: []TensorSlot,
    is_cpu: bool,
) !usize {
    var placed: std.ArrayList(usize) = .empty;
    defer placed.deinit(allocator);

    var candidates: std.ArrayList(usize) = .empty;
    defer candidates.deinit(allocator);

    var peak: usize = 0;

    for (slots) |*s| {
        if ((s.device == .cpu) != is_cpu) continue;
        if (s.birth_step > s.death_step) {
            s.arena_offset = 0;
            continue;
        }

        const s_sz = (s.byte_size + 127) & ~@as(usize, 127);
        if (s_sz == 0) {
            s.arena_offset = 0;
            continue;
        }

        // Candidates: offset 0 is always an option,
        // plus the end of any temporally conflicting placed slot.
        candidates.clearRetainingCapacity();
        try candidates.append(allocator, 0);

        for (placed.items) |p_idx| {
            const p = &slots[p_idx];
            // Intervals [b1, d1] and [b2, d2] overlap iff max(b1, b2) <= min(d1, d2)
            if (s.birth_step <= p.death_step and p.birth_step <= s.death_step) {
                const p_sz = (p.byte_size + 127) & ~@as(usize, 127);
                try candidates.append(allocator, p.arena_offset + p_sz);
            }
        }

        // Sort candidates in ascending order
        std.mem.sort(usize, candidates.items, {}, std.sort.asc(usize));

        // Find smallest candidate offset without spatial collision
        var chosen_offset: ?usize = null;
        for (candidates.items) |cand| {
            var collides = false;
            for (placed.items) |p_idx| {
                const p = &slots[p_idx];
                if (s.birth_step <= p.death_step and p.birth_step <= s.death_step) {
                    const p_sz = (p.byte_size + 127) & ~@as(usize, 127);
                    if (cand < p.arena_offset + p_sz and p.arena_offset < cand + s_sz) {
                        collides = true;
                        break;
                    }
                }
            }
            if (!collides) {
                chosen_offset = cand;
                break;
            }
        }

        const off = chosen_offset orelse peak;
        s.arena_offset = off;
        peak = @max(peak, off + s_sz);
        try placed.append(allocator, s.id);
    }

    return peak;
}

/// Ahead-Of-Time Memory Planner (zgc AOT Graph Memory Subsystem).
/// Computes lifetime intervals (birth and death step) for all intermediate activations
/// and assigns 128-byte aligned offsets into a unified scratch arena, reusing memory
/// for disjoint lifetimes with zero runtime allocations.
pub const MemoryPlanner = struct {
    pub const PlanResult = struct {
        peak_host_bytes: usize,
        peak_device_bytes: usize,
    };

    pub fn plan(
        allocator: std.mem.Allocator,
        slots: []TensorSlot,
        nodes: []const Node,
    ) !PlanResult {
        // 1. Initialize lifetimes
        for (slots) |*s| {
            s.birth_step = std.math.maxInt(usize);
            s.death_step = 0;
        }

        // 2. Compute birth and death steps from node connectivity
        for (nodes, 0..) |node, step| {
            for (node.inputs) |inp_opt| {
                if (inp_opt) |sid| {
                    if (sid < slots.len) {
                        slots[sid].birth_step = @min(slots[sid].birth_step, step);
                        slots[sid].death_step = @max(slots[sid].death_step, step);
                    }
                }
            }
            for (node.outputs) |out_opt| {
                if (out_opt) |sid| {
                    if (sid < slots.len) {
                        slots[sid].birth_step = @min(slots[sid].birth_step, step);
                        slots[sid].death_step = @max(slots[sid].death_step, step);
                    }
                }
            }
        }

        // 3. Plan host and device slots independently with interval coloring
        const peak_host = try planDevice(allocator, slots, true);
        const peak_device = try planDevice(allocator, slots, false);

        return PlanResult{
            .peak_host_bytes = peak_host,
            .peak_device_bytes = peak_device,
        };
    }
};

/// Declarative Computation Graph (DAG).
pub const ComputeGraph = struct {
    allocator: std.mem.Allocator,
    slots: std.ArrayList(TensorSlot) = .empty,
    nodes: std.ArrayList(Node) = .empty,

    pub fn init(allocator: std.mem.Allocator) ComputeGraph {
        return .{
            .allocator = allocator,
            .slots = .empty,
            .nodes = .empty,
        };
    }

    pub fn deinit(self: *ComputeGraph) void {
        self.slots.deinit(self.allocator);
        self.nodes.deinit(self.allocator);
    }

    pub fn addSlot(
        self: *ComputeGraph,
        name: []const u8,
        shape: []const usize,
        st: format.StorageType,
        device: Device,
    ) !SlotId {
        const id: SlotId = @intCast(self.slots.items.len);
        var s = TensorSlot{
            .id = id,
            .name = name,
            .storage_type = st,
            .device = device,
            .ndim = @intCast(shape.len),
            .byte_size = 0,
        };
        var numel: usize = 1;
        for (shape, 0..) |dim, i| {
            if (i < 4) s.shape[i] = dim;
            numel *= dim;
        }
        s.byte_size = numel * st.elementSize();

        try self.slots.append(self.allocator, s);
        return id;
    }

    pub fn addNode(self: *ComputeGraph, node: Node) !void {
        try self.nodes.append(self.allocator, node);
    }
};

/// Context passed to each step during graph execution.
pub const StepContext = struct {
    arena: *ActivationArena,
    pos: usize,
    token_id: u32,
    key_cache: []f32,
    val_cache: []f32,
    d_key_cache: ?*anyopaque = null,
    d_val_cache: ?*anyopaque = null,
    max_seq_len: usize = 2048,
    att: []f32 = &.{},
};

pub fn getAlignedF32Slice(bytes: []const u8, stack_buf: []f32) []const f32 {
    if (std.mem.isAligned(@intFromPtr(bytes.ptr), @alignOf(f32))) {
        const slice: []align(@alignOf(f32)) const f32 = @alignCast(std.mem.bytesAsSlice(f32, bytes));
        return slice;
    }
    const count = @min(bytes.len / 4, stack_buf.len);
    for (0..count) |i| {
        const off = i * 4;
        stack_buf[i] = @bitCast(std.mem.readInt(u32, bytes[off .. off + 4][0..4], .little));
    }
    return stack_buf[0..count];
}

/// Linearized Execution Plan compiled from ComputeGraph and MemoryPlanner.
pub const ExecutionPlan = struct {
    allocator: std.mem.Allocator,
    slots: []TensorSlot,
    nodes: []Node,
    arena: ActivationArena,
    stream: ?cuda.CudaStream = null,
    graph_exec: ?cuda.CudaGraphExec = null,

    pub fn compile(
        allocator: std.mem.Allocator,
        cg: *ComputeGraph,
    ) !ExecutionPlan {
        const planned = try MemoryPlanner.plan(allocator, cg.slots.items, cg.nodes.items);
        var arena = try ActivationArena.init(allocator, planned.peak_host_bytes, planned.peak_device_bytes);
        errdefer arena.deinit();

        const owned_slots = try allocator.dupe(TensorSlot, cg.slots.items);
        errdefer allocator.free(owned_slots);

        const owned_nodes = try allocator.dupe(Node, cg.nodes.items);
        errdefer allocator.free(owned_nodes);

        var stream: ?cuda.CudaStream = null;
        if (planned.peak_device_bytes > 0 and cuda.isAvailable()) {
            stream = cuda.CudaStream.create() catch null;
        }

        return ExecutionPlan{
            .allocator = allocator,
            .slots = owned_slots,
            .nodes = owned_nodes,
            .arena = arena,
            .stream = stream,
            .graph_exec = null,
        };
    }

    pub fn deinit(self: *ExecutionPlan) void {
        if (self.graph_exec) |*ge| ge.destroy();
        if (self.stream) |*s| s.destroy();
        self.arena.deinit();
        self.allocator.free(self.slots);
        self.allocator.free(self.nodes);
    }

    pub fn captureCudaGraph(self: *ExecutionPlan, ctx: *StepContext) !void {
        if (!cuda.enabled or !cuda.isAvailable()) return;
        const stream = self.stream orelse return;

        try cuda.CudaGraph.beginCapture(stream);
        for (self.nodes) |*node| {
            if (node.device.isCuda()) {
                try self.executeNode(node, ctx);
            }
        }
        var graph = try cuda.CudaGraph.endCapture(stream);
        defer graph.destroy();

        if (self.graph_exec) |*ge| ge.destroy();
        self.graph_exec = try graph.instantiate();
    }

    /// Executes the full computational graph for a single token decoding step.
    /// Operates entirely against pre-allocated arena offsets with 0 dynamic heap allocations.
    pub fn execute(self: *ExecutionPlan, ctx: *StepContext) !void {
        if (self.graph_exec) |*ge| {
            try ge.launch(self.stream);
            if (self.stream) |*s| try s.synchronize();
            return;
        }
        for (self.nodes) |*node| {
            try self.executeNode(node, ctx);
        }
        if (cuda.isAvailable()) {
            cuda.synchronize() catch {};
        }
    }

    fn executeNode(self: *ExecutionPlan, node: *const Node, ctx: *StepContext) !void {
        switch (node.op) {
            .embedding_lookup => {
                const out_slot = &self.slots[node.outputs[0].?];
                const out_slice = self.arena.hostSlice(out_slot);
                const w = node.weight.?;
                const dim = node.dim;
                const tid = ctx.token_id;

                switch (w.storage_type) {
                    .f32 => {
                        if (std.mem.isAligned(@intFromPtr(w.data.ptr), @alignOf(f32))) {
                            const w_f32: [*]const f32 = @ptrCast(@alignCast(w.data.ptr));
                            const row = w_f32[tid * dim .. (tid + 1) * dim];
                            @memcpy(out_slice[0..dim], row);
                        } else {
                            for (0..dim) |i| {
                                const off = (tid * dim + i) * 4;
                                const float_bytes = w.data[off .. off + 4];
                                out_slice[i] = @bitCast(std.mem.readInt(u32, float_bytes[0..4], .little));
                            }
                        }
                    },
                    .bf16 => {
                        for (0..dim) |i| {
                            const off = (tid * dim + i) * 2;
                            if (off + 2 <= w.data.len) {
                                const u_val = std.mem.readInt(u16, w.data[off .. off + 2][0..2], .little);
                                out_slice[i] = tensor_ops.bf16ToF32(u_val);
                            } else {
                                out_slice[i] = 0.0;
                            }
                        }
                    },
                    .f16 => {
                        for (0..dim) |i| {
                            const off = (tid * dim + i) * 2;
                            if (off + 2 <= w.data.len) {
                                const u_val = std.mem.readInt(u16, w.data[off .. off + 2][0..2], .little);
                                out_slice[i] = @floatCast(@as(f16, @bitCast(u_val)));
                            } else {
                                out_slice[i] = 0.0;
                            }
                        }
                    },
                    .q8_0 => {
                        const blocks_per_row = dim / 32;
                        const row_bytes = blocks_per_row * 34;
                        const row_slice = w.data[tid * row_bytes .. (tid + 1) * row_bytes];
                        for (0..blocks_per_row) |b| {
                            const blk_bytes = row_slice[b * 34 .. (b + 1) * 34];
                            const d_raw = std.mem.readInt(u16, blk_bytes[0..2], .little);
                            const d: f32 = @floatCast(@as(f16, @bitCast(d_raw)));
                            for (0..32) |i| {
                                const q: i8 = @bitCast(blk_bytes[2 + i]);
                                out_slice[b * 32 + i] = @as(f32, @floatFromInt(q)) * d;
                            }
                        }
                    },
                    .q4_0 => {
                        const blocks_per_row = dim / 32;
                        const row_bytes = blocks_per_row * 18;
                        const row_slice = w.data[tid * row_bytes .. (tid + 1) * row_bytes];
                        for (0..blocks_per_row) |b| {
                            const blk_bytes = row_slice[b * 18 .. (b + 1) * 18];
                            const d_raw = std.mem.readInt(u16, blk_bytes[0..2], .little);
                            const d: f32 = @floatCast(@as(f16, @bitCast(d_raw)));
                            for (0..16) |i| {
                                const byte = blk_bytes[2 + i];
                                const q0: i8 = @as(i8, @intCast(byte & 0x0F)) - 8;
                                const q1: i8 = @as(i8, @intCast((byte >> 4) & 0x0F)) - 8;
                                out_slice[b * 32 + i] = @as(f32, @floatFromInt(q0)) * d;
                                out_slice[b * 32 + i + 16] = @as(f32, @floatFromInt(q1)) * d;
                            }
                        }
                    },
                    else => {
                        @memset(out_slice[0..dim], 0.0);
                    },
                }
            },
            .rmsnorm => {
                const in_slot = &self.slots[node.inputs[0].?];
                const out_slot = &self.slots[node.outputs[0].?];
                const w = node.weight.?;

                if (node.device.isCuda()) {
                    const d_in = self.arena.devicePtr(in_slot).?;
                    const d_out = self.arena.devicePtr(out_slot).?;
                    const d_w: *const anyopaque = @ptrCast(w.gpu_buf.?.ptr);
                    _ = cuda.rmsNorm(d_in, d_w, d_out, @intCast(node.dim), node.eps) catch {};
                } else {
                    const in_slice = self.arena.hostSliceConst(in_slot);
                    const out_slice = self.arena.hostSlice(out_slot);
                    var stack_w: [4096]f32 = undefined;
                    const w_slice = getAlignedF32Slice(w.data, &stack_w);
                    tensor_ops.rmsNormF32(in_slice, w_slice, node.eps, out_slice);
                }
            },
            .head_rmsnorm => {
                const slot = &self.slots[node.inputs[0].?];
                const w = node.weight.?;

                if (node.device.isCuda()) {
                    const d_ptr = self.arena.devicePtr(slot).?;
                    const d_w: *const anyopaque = @ptrCast(w.gpu_buf.?.ptr);
                    _ = cuda.headRmsNorm(d_ptr, d_w, @intCast(node.n_heads), @intCast(node.head_dim), @intCast(w.rows), node.eps) catch {};
                } else {
                    const slice = self.arena.hostSlice(slot);
                    var stack_w: [4096]f32 = undefined;
                    const w_slice = getAlignedF32Slice(w.data, &stack_w);
                    const head_dim = node.head_dim;
                    for (0..node.n_heads) |h| {
                        const h_off = h * head_dim;
                        const head_slice = slice[h_off .. h_off + head_dim];
                        const w_head = if (w.rows == head_dim) w_slice[0..head_dim] else w_slice[h_off .. h_off + head_dim];
                        tensor_ops.rmsNormF32(head_slice, w_head, node.eps, head_slice);
                    }
                }
            },
            .rope => {
                const q_slot = &self.slots[node.inputs[0].?];
                const k_slot = &self.slots[node.inputs[1].?];

                if (node.device.isCuda()) {
                    const d_q = self.arena.devicePtr(q_slot).?;
                    const d_k = self.arena.devicePtr(k_slot).?;
                    _ = cuda.rope(d_q, d_k, @intCast(ctx.pos), @intCast(node.n_heads), @intCast(node.n_kv_heads), @intCast(node.head_dim), node.rope_theta) catch {};
                } else {
                    const q_slice = self.arena.hostSlice(q_slot);
                    const k_slice = self.arena.hostSlice(k_slot);
                    const head_dim = node.head_dim;
                    const half_dim = head_dim / 2;

                    for (0..node.n_heads) |h| {
                        if ((h + 1) * head_dim > q_slice.len) break;
                        const q_head = q_slice[h * head_dim .. (h + 1) * head_dim];
                        for (0..half_dim) |i| {
                            const freq = 1.0 / std.math.pow(f32, node.rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
                            const val = @as(f32, @floatFromInt(ctx.pos)) * freq;
                            const cos_val = @cos(val);
                            const sin_val = @sin(val);
                            const v0 = q_head[2 * i];
                            const v1 = q_head[2 * i + 1];
                            q_head[2 * i] = v0 * cos_val - v1 * sin_val;
                            q_head[2 * i + 1] = v0 * sin_val + v1 * cos_val;
                        }
                    }

                    for (0..node.n_kv_heads) |h| {
                        if ((h + 1) * head_dim > k_slice.len) break;
                        const k_head = k_slice[h * head_dim .. (h + 1) * head_dim];
                        for (0..half_dim) |i| {
                            const freq = 1.0 / std.math.pow(f32, node.rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
                            const val = @as(f32, @floatFromInt(ctx.pos)) * freq;
                            const cos_val = @cos(val);
                            const sin_val = @sin(val);
                            const v0 = k_head[2 * i];
                            const v1 = k_head[2 * i + 1];
                            k_head[2 * i] = v0 * cos_val - v1 * sin_val;
                            k_head[2 * i + 1] = v0 * sin_val + v1 * cos_val;
                        }
                    }
                }
            },
            .gemv => {
                const in_slot = &self.slots[node.inputs[0].?];
                const out_slot = &self.slots[node.outputs[0].?];
                const w = node.weight.?;

                if (node.device.isCuda()) {
                    const d_in = self.arena.devicePtr(in_slot).?;
                    const d_out = self.arena.devicePtr(out_slot).?;
                    const d_w: *const anyopaque = @ptrCast(w.gpu_buf.?.ptr);
                    switch (w.storage_type) {
                        .f32 => _ = cuda.gemvF32(d_w, d_in, d_out, @intCast(w.rows), @intCast(w.cols)) catch {},
                        .q8_0 => _ = cuda.gemvQ8_0(d_w, d_in, d_out, @intCast(w.rows), @intCast(w.cols)) catch {},
                        .q4_0 => _ = cuda.gemvQ4_0(d_w, d_in, d_out, @intCast(w.rows), @intCast(w.cols)) catch {},
                        else => {},
                    }
                } else {
                    const in_slice = self.arena.hostSliceConst(in_slot);
                    const out_slice = self.arena.hostSlice(out_slot);
                    matVec(w, in_slice, out_slice);
                }
            },
            .attention_gqa => {
                const q_slot = &self.slots[node.inputs[0].?];
                const k_slot = &self.slots[node.inputs[1].?];
                const v_slot = &self.slots[node.inputs[2].?];
                const out_slot = &self.slots[node.outputs[0].?];

                const kv_dim = @max(node.n_kv_heads * node.head_dim, node.dim);
                const layer_off = node.layer_idx * ctx.max_seq_len * kv_dim;
                const pos_off = layer_off + ctx.pos * kv_dim;

                if (node.device.isCuda()) {
                    const d_q = self.arena.devicePtr(q_slot).?;
                    const d_out = self.arena.devicePtr(out_slot).?;
                    const d_k = self.arena.devicePtr(k_slot).?;
                    const d_v = self.arena.devicePtr(v_slot).?;

                    _ = cuda.kvCacheUpdate(ctx.d_key_cache.?, ctx.d_val_cache.?, d_k, d_v, @intCast(node.layer_idx), @intCast(ctx.pos), @intCast(ctx.max_seq_len), @intCast(kv_dim)) catch {};
                    _ = cuda.gqaAttention(d_q, ctx.d_key_cache.?, ctx.d_val_cache.?, d_out, @intCast(node.layer_idx), @intCast(ctx.pos), @intCast(node.n_heads), @intCast(node.n_kv_heads), @intCast(node.head_dim), @intCast(ctx.max_seq_len), @intCast(kv_dim)) catch {};
                } else {
                    const q_slice = self.arena.hostSliceConst(q_slot);
                    const k_slice = self.arena.hostSliceConst(k_slot);
                    const v_slice = self.arena.hostSliceConst(v_slot);
                    const out_slice = self.arena.hostSlice(out_slot);

                    @memcpy(ctx.key_cache[pos_off .. pos_off + kv_dim], k_slice[0..kv_dim]);
                    @memcpy(ctx.val_cache[pos_off .. pos_off + kv_dim], v_slice[0..kv_dim]);

                    @memset(out_slice, 0.0);
                    const head_dim = node.head_dim;
                    const n_heads = node.n_heads;
                    const n_kv_heads = node.n_kv_heads;
                    const n_rep = if (n_kv_heads > 0) @max(1, n_heads / n_kv_heads) else 1;
                    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(@max(head_dim, 1))));
                    const max_t = @min(ctx.pos + 1, ctx.max_seq_len);

                    var att_stack: [2048]f32 = undefined;
                    const att = if (ctx.att.len >= max_t) ctx.att[0..max_t] else att_stack[0..@min(max_t, 2048)];

                    for (0..n_heads) |h| {
                        if ((h + 1) * head_dim > q_slice.len or (h + 1) * head_dim > out_slice.len) break;
                        const q_head = q_slice[h * head_dim .. (h + 1) * head_dim];
                        const kv_head_idx = h / n_rep;

                        for (0..max_t) |t| {
                            const t_off = layer_off + t * kv_dim + kv_head_idx * head_dim;
                            const k_head = ctx.key_cache[t_off .. t_off + head_dim];
                            att[t] = tensor_ops.dotProductF32(q_head, k_head) * scale;
                        }

                        tensor_ops.softmaxF32(att, att);

                        const out_head = out_slice[h * head_dim .. (h + 1) * head_dim];
                        for (0..max_t) |t| {
                            const a = att[t];
                            const t_off = layer_off + t * kv_dim + kv_head_idx * head_dim;
                            const v_head = ctx.val_cache[t_off .. t_off + head_dim];
                            for (0..head_dim) |d| {
                                out_head[d] += a * v_head[d];
                            }
                        }
                    }
                }
            },
            .swiglu => {
                const gate_slot = &self.slots[node.inputs[0].?];
                const up_slot = &self.slots[node.inputs[1].?];

                if (node.device.isCuda()) {
                    const d_gate = self.arena.devicePtr(gate_slot).?;
                    const d_up = self.arena.devicePtr(up_slot).?;
                    _ = cuda.swiglu(d_gate, d_up, @intCast(node.hidden_dim)) catch {};
                } else {
                    const gate_slice = self.arena.hostSlice(gate_slot);
                    const up_slice = self.arena.hostSliceConst(up_slot);
                    for (0..node.hidden_dim) |i| {
                        const g = gate_slice[i];
                        const silu_g = g / (1.0 + @exp(-g));
                        gate_slice[i] = silu_g * up_slice[i];
                    }
                }
            },
            .add_residual => {
                const x_slot = &self.slots[node.inputs[0].?];
                const res_slot = &self.slots[node.inputs[1].?];

                if (node.device.isCuda()) {
                    const d_x = self.arena.devicePtr(x_slot).?;
                    const d_res = self.arena.devicePtr(res_slot).?;
                    _ = cuda.addResidual(d_x, d_res, @intCast(node.dim)) catch {};
                } else {
                    const x_slice = self.arena.hostSlice(x_slot);
                    const res_slice = self.arena.hostSliceConst(res_slot);
                    for (0..node.dim) |i| {
                        x_slice[i] += res_slice[i];
                    }
                }
            },
            .device_copy_h2d => {
                const host_slot = &self.slots[node.inputs[0].?];
                const dev_slot = &self.slots[node.outputs[0].?];
                const host_slice = self.arena.hostSliceConst(host_slot);
                const dev_ptr = self.arena.devicePtr(dev_slot).?;
                _ = cuda.uploadTo(dev_ptr, std.mem.sliceAsBytes(host_slice)) catch {};
            },
            .device_copy_d2h => {
                const dev_slot = &self.slots[node.inputs[0].?];
                const host_slot = &self.slots[node.outputs[0].?];
                const dev_ptr = self.arena.devicePtr(dev_slot).?;
                const host_slice = self.arena.hostSlice(host_slot);
                _ = cuda.downloadFrom(std.mem.sliceAsBytes(host_slice), dev_ptr) catch {};
            },
            .fused_qknorm_rope => {
                const q_slot = &self.slots[node.inputs[0].?];
                const k_slot = &self.slots[node.inputs[1].?];
                const w_q = node.weight.?;
                const w_k = node.aux_weight.?;

                if (node.device.isCuda()) {
                    const d_q = self.arena.devicePtr(q_slot).?;
                    const d_k = self.arena.devicePtr(k_slot).?;
                    const d_wq: ?*const anyopaque = if (w_q.gpu_buf) |b| @ptrCast(b.ptr) else null;
                    const d_wk: ?*const anyopaque = if (w_k.gpu_buf) |b| @ptrCast(b.ptr) else null;
                    _ = cuda.fusedQkNormRope(d_q, d_k, d_wq, d_wk, @intCast(ctx.pos), @intCast(node.n_heads), @intCast(node.n_kv_heads), @intCast(node.head_dim), @intCast(w_q.rows), @intCast(w_k.rows), node.eps, node.rope_theta) catch {};
                } else {
                    const q_slice = self.arena.hostSlice(q_slot);
                    const k_slice = self.arena.hostSlice(k_slot);
                    var stack_wq: [4096]f32 = undefined;
                    var stack_wk: [4096]f32 = undefined;
                    const wq_slice = getAlignedF32Slice(w_q.data, &stack_wq);
                    const wk_slice = getAlignedF32Slice(w_k.data, &stack_wk);
                    const head_dim = node.head_dim;

                    for (0..node.n_heads) |h| {
                        const h_off = h * head_dim;
                        const head_slice = q_slice[h_off .. h_off + head_dim];
                        const w_head = if (w_q.rows == head_dim) wq_slice[0..head_dim] else wq_slice[h_off .. h_off + head_dim];
                        tensor_ops.rmsNormF32(head_slice, w_head, node.eps, head_slice);
                    }

                    for (0..node.n_kv_heads) |h| {
                        const h_off = h * head_dim;
                        const head_slice = k_slice[h_off .. h_off + head_dim];
                        const w_head = if (w_k.rows == head_dim) wk_slice[0..head_dim] else wk_slice[h_off .. h_off + head_dim];
                        tensor_ops.rmsNormF32(head_slice, w_head, node.eps, head_slice);
                    }

                    const half_dim = head_dim / 2;
                    for (0..node.n_heads) |h| {
                        if ((h + 1) * head_dim > q_slice.len) break;
                        const q_head = q_slice[h * head_dim .. (h + 1) * head_dim];
                        for (0..half_dim) |i| {
                            const freq = 1.0 / std.math.pow(f32, node.rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
                            const val = @as(f32, @floatFromInt(ctx.pos)) * freq;
                            const cos_val = @cos(val);
                            const sin_val = @sin(val);
                            const v0 = q_head[2 * i];
                            const v1 = q_head[2 * i + 1];
                            q_head[2 * i] = v0 * cos_val - v1 * sin_val;
                            q_head[2 * i + 1] = v0 * sin_val + v1 * cos_val;
                        }
                    }

                    for (0..node.n_kv_heads) |h| {
                        if ((h + 1) * head_dim > k_slice.len) break;
                        const k_head = k_slice[h * head_dim .. (h + 1) * head_dim];
                        for (0..half_dim) |i| {
                            const freq = 1.0 / std.math.pow(f32, node.rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
                            const val = @as(f32, @floatFromInt(ctx.pos)) * freq;
                            const cos_val = @cos(val);
                            const sin_val = @sin(val);
                            const v0 = k_head[2 * i];
                            const v1 = k_head[2 * i + 1];
                            k_head[2 * i] = v0 * cos_val - v1 * sin_val;
                            k_head[2 * i + 1] = v0 * sin_val + v1 * cos_val;
                        }
                    }
                }
            },
            .fused_swiglu_residual => {
                const gate_slot = &self.slots[node.inputs[0].?];
                const up_slot = &self.slots[node.inputs[1].?];
                const res_slot = if (node.inputs[2]) |sid| &self.slots[sid] else null;

                if (node.device.isCuda()) {
                    const d_gate = self.arena.devicePtr(gate_slot).?;
                    const d_up = self.arena.devicePtr(up_slot).?;
                    const d_res: ?*const anyopaque = if (res_slot) |s| @ptrCast(self.arena.devicePtr(s).?) else null;
                    _ = cuda.fusedSwiGluResidual(d_gate, d_up, d_res, @intCast(node.hidden_dim), @intCast(node.dim)) catch {};
                } else {
                    const gate_slice = self.arena.hostSlice(gate_slot);
                    const up_slice = self.arena.hostSliceConst(up_slot);
                    for (0..node.hidden_dim) |i| {
                        const g = gate_slice[i];
                        const silu_g = g / (1.0 + @exp(-g));
                        gate_slice[i] = silu_g * up_slice[i];
                    }
                    if (res_slot) |s| {
                        const res_slice = self.arena.hostSliceConst(s);
                        for (0..node.dim) |i| {
                            gate_slice[i] += res_slice[i];
                        }
                    }
                }
            },
        }
    }
};

pub fn matVec(weight: WeightRef, in_x: []const f32, out_y: []f32) void {
    const safe_rows = @min(weight.rows, out_y.len);
    switch (weight.storage_type) {
        .q8_0 => tensor_ops.gemvQ8_0(weight.data, in_x, null, out_y[0..safe_rows], safe_rows, weight.cols),
        .q4_0 => tensor_ops.gemvQ4_0(weight.data, in_x, null, out_y[0..safe_rows], safe_rows, weight.cols),
        .q4_k => tensor_ops.gemvQ4_K(weight.data, in_x, null, out_y[0..safe_rows], safe_rows, weight.cols),
        .f32 => {
            if (std.mem.isAligned(@intFromPtr(weight.data.ptr), @alignOf(f32))) {
                const w_f32: [*]const f32 = @ptrCast(@alignCast(weight.data.ptr));
                tensor_ops.gemvF32(w_f32[0 .. weight.rows * weight.cols], in_x, null, out_y[0..safe_rows], safe_rows, weight.cols);
            } else {
                for (0..safe_rows) |r| {
                    var dot: f32 = 0.0;
                    const row_offset = r * weight.cols * 4;
                    for (0..weight.cols) |c| {
                        const float_bytes = weight.data[row_offset + c * 4 .. row_offset + (c + 1) * 4];
                        const f_val: f32 = @bitCast(std.mem.readInt(u32, float_bytes[0..4], .little));
                        dot += f_val * in_x[c];
                    }
                    out_y[r] = dot;
                }
            }
        },
        .bf16 => {
            if (std.mem.isAligned(@intFromPtr(weight.data.ptr), @alignOf(u16))) {
                const w_u16: [*]const u16 = @ptrCast(@alignCast(weight.data.ptr));
                const total = @min(weight.rows * weight.cols, weight.data.len / 2);
                tensor_ops.gemvBF16(w_u16[0..total], in_x, null, out_y[0..safe_rows], safe_rows, weight.cols);
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
                out_y[r] = dot;
            }
        },
        else => {
            if (std.mem.isAligned(@intFromPtr(weight.data.ptr), @alignOf(f32))) {
                const w_f32: [*]const f32 = @ptrCast(@alignCast(weight.data.ptr));
                tensor_ops.gemvF32(w_f32[0 .. weight.rows * weight.cols], in_x, null, out_y[0..safe_rows], safe_rows, weight.cols);
            }
        },
    }
}

// ---------------------------------------------------------------------
// Unit Tests (zgc Memory Planning & Execution Verification)
// ---------------------------------------------------------------------
test "MemoryPlanner: disjoint lifetimes reuse arena offset" {
    const allocator = std.testing.allocator;
    var cg = ComputeGraph.init(allocator);
    defer cg.deinit();

    // Slot 0: used in step 0, dead after step 1
    const s0 = try cg.addSlot("slot0", &.{64}, .f32, .cpu);
    // Slot 1: born at step 2, dead after step 3 (can reuse slot0!)
    const s1 = try cg.addSlot("slot1", &.{64}, .f32, .cpu);
    // Slot 2: born at step 0, lives until step 4 (overlaps both, cannot reuse)
    const s2 = try cg.addSlot("slot2", &.{64}, .f32, .cpu);

    try cg.addNode(.{ .op = .rmsnorm, .name = "n0", .inputs = .{ null, null, null }, .outputs = .{ s0, s2 }, .dim = 64 });
    try cg.addNode(.{ .op = .rmsnorm, .name = "n1", .inputs = .{ s0, null, null }, .outputs = .{ null, null }, .dim = 64 });
    try cg.addNode(.{ .op = .rmsnorm, .name = "n2", .inputs = .{ null, null, null }, .outputs = .{ s1, null }, .dim = 64 });
    try cg.addNode(.{ .op = .rmsnorm, .name = "n3", .inputs = .{ s1, s2, null }, .outputs = .{ null, null }, .dim = 64 });

    var plan = try ExecutionPlan.compile(allocator, &cg);
    defer plan.deinit();

    // Verify slot0 and slot1 reuse the exact same arena offset
    try std.testing.expectEqual(plan.slots[s0].arena_offset, plan.slots[s1].arena_offset);
    // Verify slot2 gets a distinct non-overlapping offset
    try std.testing.expect(plan.slots[s2].arena_offset != plan.slots[s0].arena_offset);
}

test "ExecutionPlan: RMSNorm, SwiGLU, and Residual execution correctness" {
    const allocator = std.testing.allocator;
    var cg = ComputeGraph.init(allocator);
    defer cg.deinit();

    const s_in = try cg.addSlot("in", &.{4}, .f32, .cpu);
    const s_norm = try cg.addSlot("norm", &.{4}, .f32, .cpu);
    const s_up = try cg.addSlot("up", &.{4}, .f32, .cpu);

    const norm_w = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    try cg.addNode(.{
        .op = .rmsnorm,
        .name = "test_norm",
        .inputs = .{ s_in, null, null },
        .outputs = .{ s_norm, null },
        .weight = .{
            .data = std.mem.sliceAsBytes(&norm_w),
            .storage_type = .f32,
            .rows = 4,
            .cols = 1,
        },
        .dim = 4,
        .eps = 1e-5,
    });

    try cg.addNode(.{
        .op = .swiglu,
        .name = "test_swiglu",
        .inputs = .{ s_norm, s_up, null },
        .outputs = .{ s_norm, null },
        .hidden_dim = 4,
    });

    var plan = try ExecutionPlan.compile(allocator, &cg);
    defer plan.deinit();

    // Populate inputs in arena
    const in_slice = plan.arena.hostSlice(&plan.slots[s_in]);
    @memcpy(in_slice, &[_]f32{ 2.0, -2.0, 2.0, -2.0 });

    const up_slice = plan.arena.hostSlice(&plan.slots[s_up]);
    @memcpy(up_slice, &[_]f32{ 1.0, 2.0, 0.5, 1.0 });

    var dummy_key = [_]f32{0} ** 16;
    var dummy_val = [_]f32{0} ** 16;
    var ctx = StepContext{
        .arena = &plan.arena,
        .pos = 0,
        .token_id = 0,
        .key_cache = &dummy_key,
        .val_cache = &dummy_val,
    };

    try plan.execute(&ctx);

    const norm_slice = plan.arena.hostSlice(&plan.slots[s_norm]);
    // RMS of [2, -2, 2, -2] is sqrt(4) = 2.0.
    // Normalized is [1.0, -1.0, 1.0, -1.0].
    // SiLU(1.0) = 1.0 / (1 + exp(-1)) ~= 0.7310586
    // SwiGLU out[0] = 0.7310586 * 1.0 = 0.7310586
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), norm_slice[0], 0.01);
}

