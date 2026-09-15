//! Zig bindings for the CUDA GPU backend (src/cuda/hk_cuda.cu).
//!
//! Architected on PyTorch CUDA execution principles:
//! 1. CUDA Streams (asynchronous non-blocking kernel queue execution)
//! 2. CUDA Events (stream synchronization, timing, GPU elapsed measurement)
//! 3. CUDA Graphs & GraphExec (whole-graph hardware capture & single-launch replay)
//! 4. CUDA Binned Caching Allocator (small <= 1MB, large > 1MB pools for zero-alloc inference)
//! 5. Unified DeviceBuffer & Raw Pointer interoperability with automatic type coercion
//!
//! Only linked in when built with `-Dcuda=true`; otherwise every function
//! returns error.CudaNotCompiledIn so callers can branch on
//! `cuda.enabled` at comptime or try calls gracefully.

const std = @import("std");
const build_options = @import("build_options");
const format = @import("format.zig");

pub const enabled = build_options.cuda;

pub const CudaError = error{
    CudaNotCompiledIn,
    CudaCallFailed,
    NoCudaDevice,
    UnsupportedStorageType,
};

const c = if (enabled) struct {
    extern fn hk_cuda_device_count() c_int;
    extern fn hk_cuda_get_device_name(buf: [*]u8, buf_len: c_int) c_int;
    extern fn hk_cuda_device_total_mem() c_longlong;
    extern fn hk_cuda_device_free_mem() c_longlong;
    extern fn hk_cuda_malloc(nbytes: usize) ?*anyopaque;
    extern fn hk_cuda_upload(dev_ptr: ?*anyopaque, host_ptr: ?*const anyopaque, nbytes: usize) c_int;
    extern fn hk_cuda_download(host_ptr: ?*anyopaque, dev_ptr: ?*const anyopaque, nbytes: usize) c_int;
    extern fn hk_cuda_free(dev_ptr: ?*anyopaque) void;
    extern fn hk_cuda_synchronize() c_int;

    // Streams & Events
    extern fn hk_cuda_stream_create(p_stream: *?*anyopaque) c_int;
    extern fn hk_cuda_stream_destroy(stream: ?*anyopaque) c_int;
    extern fn hk_cuda_stream_synchronize(stream: ?*anyopaque) c_int;
    extern fn hk_cuda_event_create(p_event: *?*anyopaque) c_int;
    extern fn hk_cuda_event_destroy(event: ?*anyopaque) c_int;
    extern fn hk_cuda_event_record(event: ?*anyopaque, stream: ?*anyopaque) c_int;
    extern fn hk_cuda_event_synchronize(event: ?*anyopaque) c_int;
    extern fn hk_cuda_event_elapsed_ms(start: ?*anyopaque, end: ?*anyopaque) f32;

    // Hardware Graphs
    extern fn hk_cuda_graph_begin_capture(stream: ?*anyopaque) c_int;
    extern fn hk_cuda_graph_end_capture(stream: ?*anyopaque, p_graph: *?*anyopaque) c_int;
    extern fn hk_cuda_graph_instantiate(p_exec: *?*anyopaque, graph: ?*anyopaque) c_int;
    extern fn hk_cuda_graph_launch(exec: ?*anyopaque, stream: ?*anyopaque) c_int;
    extern fn hk_cuda_graph_destroy(graph: ?*anyopaque) c_int;
    extern fn hk_cuda_graph_exec_destroy(exec: ?*anyopaque) c_int;

    // Kernel Dispatches
    extern fn hk_cuda_gemv_f32(w: ?*const anyopaque, x: ?*const anyopaque, y: ?*anyopaque, m: c_int, k: c_int) c_int;
    extern fn hk_cuda_gemv_q8_0(w: ?*const anyopaque, x: ?*const anyopaque, y: ?*anyopaque, m: c_int, k: c_int) c_int;
    extern fn hk_cuda_gemv_q4_0(w: ?*const anyopaque, x: ?*const anyopaque, y: ?*anyopaque, m: c_int, k: c_int) c_int;
    extern fn hk_cuda_rmsnorm(x: ?*const anyopaque, weight: ?*const anyopaque, out: ?*anyopaque, dim: c_int, eps: f32) c_int;
    extern fn hk_cuda_head_rmsnorm(x: ?*anyopaque, weight: ?*const anyopaque, n_heads: c_int, head_dim: c_int, weight_len: c_int, eps: f32) c_int;
    extern fn hk_cuda_rope(q: ?*anyopaque, k: ?*anyopaque, pos: c_int, n_heads: c_int, n_kv_heads: c_int, head_dim: c_int, rope_theta: f32) c_int;
    extern fn hk_cuda_kv_cache_update(key_cache: ?*anyopaque, val_cache: ?*anyopaque, k: ?*const anyopaque, v: ?*const anyopaque, layer: c_int, pos: c_int, max_seq_len: c_int, kv_dim: c_int) c_int;
    extern fn hk_cuda_gqa_attention(q: ?*const anyopaque, key_cache: ?*const anyopaque, val_cache: ?*const anyopaque, out: ?*anyopaque, layer: c_int, pos: c_int, n_heads: c_int, n_kv_heads: c_int, head_dim: c_int, max_seq_len: c_int, kv_dim: c_int) c_int;
    extern fn hk_cuda_swiglu(gate: ?*anyopaque, up: ?*const anyopaque, hidden_dim: c_int) c_int;
    extern fn hk_cuda_add_residual(x: ?*anyopaque, residual: ?*const anyopaque, dim: c_int) c_int;

    // Fused Kernels
    extern fn hk_cuda_fused_qknorm_rope(q: ?*anyopaque, k: ?*anyopaque, wq: ?*const anyopaque, wk: ?*const anyopaque, pos: c_int, n_heads: c_int, n_kv_heads: c_int, head_dim: c_int, w_len_q: c_int, w_len_k: c_int, eps: f32, rope_theta: f32) c_int;
    extern fn hk_cuda_fused_swiglu_residual(gate: ?*anyopaque, up: ?*const anyopaque, residual: ?*const anyopaque, hidden_dim: c_int, dim: c_int) c_int;
} else struct {};

pub fn deviceCount() i32 {
    if (!enabled) return 0;
    return @intCast(c.hk_cuda_device_count());
}

pub fn isAvailable() bool {
    return deviceCount() > 0;
}

pub fn getDeviceName(buf: []u8) ![]const u8 {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_get_device_name(buf.ptr, @intCast(buf.len));
    if (rc != 0) return CudaError.CudaCallFailed;
    return std_mem_sliceTo(buf, 0);
}

fn std_mem_sliceTo(buf: []u8, sentinel: u8) []const u8 {
    var i: usize = 0;
    while (i < buf.len and buf[i] != sentinel) : (i += 1) {}
    return buf[0..i];
}

pub fn totalMemBytes() !i64 {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const v = c.hk_cuda_device_total_mem();
    if (v < 0) return CudaError.CudaCallFailed;
    return v;
}

pub fn freeMemBytes() !i64 {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const v = c.hk_cuda_device_free_mem();
    if (v < 0) return CudaError.CudaCallFailed;
    return v;
}

// ---------------------------------------------------------------------
// PyTorch-style Binned Caching Allocator
// ---------------------------------------------------------------------
pub const CudaCachingAllocator = struct {
    pub const Block = struct {
        ptr: *anyopaque,
        size: usize,
    };

    var mutex: std.Thread.Mutex = .{};
    var small_blocks: [128]?Block = [_]?Block{null} ** 128;
    var small_count: usize = 0;
    var large_blocks: [64]?Block = [_]?Block{null} ** 64;
    var large_count: usize = 0;

    pub const SMALL_THRESHOLD: usize = 1024 * 1024; // 1 MB

    pub fn alloc(nbytes: usize) !*anyopaque {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        mutex.lock();
        defer mutex.unlock();

        if (nbytes <= SMALL_THRESHOLD) {
            var best_idx: ?usize = null;
            var best_diff: usize = std.math.maxInt(usize);
            for (0..small_count) |i| {
                if (small_blocks[i]) |b| {
                    if (b.size >= nbytes and (b.size - nbytes) < best_diff) {
                        best_diff = b.size - nbytes;
                        best_idx = i;
                    }
                }
            }
            if (best_idx) |idx| {
                const blk = small_blocks[idx].?;
                small_blocks[idx] = small_blocks[small_count - 1];
                small_blocks[small_count - 1] = null;
                small_count -= 1;
                return blk.ptr;
            }
        } else {
            var best_idx: ?usize = null;
            var best_diff: usize = std.math.maxInt(usize);
            for (0..large_count) |i| {
                if (large_blocks[i]) |b| {
                    if (b.size >= nbytes and (b.size - nbytes) < best_diff) {
                        best_diff = b.size - nbytes;
                        best_idx = i;
                    }
                }
            }
            if (best_idx) |idx| {
                const blk = large_blocks[idx].?;
                large_blocks[idx] = large_blocks[large_count - 1];
                large_blocks[large_count - 1] = null;
                large_count -= 1;
                return blk.ptr;
            }
        }

        const dev_ptr = c.hk_cuda_malloc(nbytes) orelse return CudaError.CudaCallFailed;
        return dev_ptr;
    }

    pub fn free(ptr: *anyopaque, size: usize) void {
        if (!enabled) return;
        mutex.lock();
        defer mutex.unlock();

        if (size <= SMALL_THRESHOLD and small_count < small_blocks.len) {
            small_blocks[small_count] = .{ .ptr = ptr, .size = size };
            small_count += 1;
            return;
        } else if (size > SMALL_THRESHOLD and large_count < large_blocks.len) {
            large_blocks[large_count] = .{ .ptr = ptr, .size = size };
            large_count += 1;
            return;
        }

        c.hk_cuda_free(ptr);
    }

    pub fn emptyCache() void {
        if (!enabled) return;
        mutex.lock();
        defer mutex.unlock();

        for (0..small_count) |i| {
            if (small_blocks[i]) |b| {
                c.hk_cuda_free(b.ptr);
                small_blocks[i] = null;
            }
        }
        small_count = 0;

        for (0..large_count) |i| {
            if (large_blocks[i]) |b| {
                c.hk_cuda_free(b.ptr);
                large_blocks[i] = null;
            }
        }
        large_count = 0;
    }
};

pub fn emptyCache() void {
    CudaCachingAllocator.emptyCache();
}

// ---------------------------------------------------------------------
// CUDA Streams & Events
// ---------------------------------------------------------------------
pub const CudaStream = struct {
    handle: *anyopaque,

    pub fn create() !CudaStream {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        var h: ?*anyopaque = null;
        if (c.hk_cuda_stream_create(&h) != 0 or h == null) {
            return CudaError.CudaCallFailed;
        }
        return CudaStream{ .handle = h.? };
    }

    pub fn destroy(self: *CudaStream) void {
        if (!enabled) return;
        _ = c.hk_cuda_stream_destroy(self.handle);
    }

    pub fn synchronize(self: *const CudaStream) !void {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        if (c.hk_cuda_stream_synchronize(self.handle) != 0) {
            return CudaError.CudaCallFailed;
        }
    }
};

pub const CudaEvent = struct {
    handle: *anyopaque,

    pub fn create() !CudaEvent {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        var h: ?*anyopaque = null;
        if (c.hk_cuda_event_create(&h) != 0 or h == null) {
            return CudaError.CudaCallFailed;
        }
        return CudaEvent{ .handle = h.? };
    }

    pub fn destroy(self: *CudaEvent) void {
        if (!enabled) return;
        _ = c.hk_cuda_event_destroy(self.handle);
    }

    pub fn record(self: *const CudaEvent, stream: ?CudaStream) !void {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        const s_handle = if (stream) |s| s.handle else null;
        if (c.hk_cuda_event_record(self.handle, s_handle) != 0) {
            return CudaError.CudaCallFailed;
        }
    }

    pub fn synchronize(self: *const CudaEvent) !void {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        if (c.hk_cuda_event_synchronize(self.handle) != 0) {
            return CudaError.CudaCallFailed;
        }
    }

    pub fn elapsedMs(start: *const CudaEvent, end: *const CudaEvent) !f32 {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        return c.hk_cuda_event_elapsed_ms(start.handle, end.handle);
    }
};

// ---------------------------------------------------------------------
// CUDA Graphs (Capture & Replay)
// ---------------------------------------------------------------------
pub const CudaGraph = struct {
    handle: *anyopaque,

    pub fn beginCapture(stream: CudaStream) !void {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        if (c.hk_cuda_graph_begin_capture(stream.handle) != 0) {
            return CudaError.CudaCallFailed;
        }
    }

    pub fn endCapture(stream: CudaStream) !CudaGraph {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        var h: ?*anyopaque = null;
        if (c.hk_cuda_graph_end_capture(stream.handle, &h) != 0 or h == null) {
            return CudaError.CudaCallFailed;
        }
        return CudaGraph{ .handle = h.? };
    }

    pub fn instantiate(self: *const CudaGraph) !CudaGraphExec {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        var exec_h: ?*anyopaque = null;
        if (c.hk_cuda_graph_instantiate(&exec_h, self.handle) != 0 or exec_h == null) {
            return CudaError.CudaCallFailed;
        }
        return CudaGraphExec{ .handle = exec_h.? };
    }

    pub fn destroy(self: *CudaGraph) void {
        if (!enabled) return;
        _ = c.hk_cuda_graph_destroy(self.handle);
    }
};

pub const CudaGraphExec = struct {
    handle: *anyopaque,

    pub fn launch(self: *const CudaGraphExec, stream: ?CudaStream) !void {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        const s_handle = if (stream) |s| s.handle else null;
        if (c.hk_cuda_graph_launch(self.handle, s_handle) != 0) {
            return CudaError.CudaCallFailed;
        }
    }

    pub fn destroy(self: *CudaGraphExec) void {
        if (!enabled) return;
        _ = c.hk_cuda_graph_exec_destroy(self.handle);
    }
};

// ---------------------------------------------------------------------
// Owning Handle to a Device Allocation
// ---------------------------------------------------------------------
pub const DeviceBuffer = struct {
    ptr: *anyopaque,
    len: usize,

    pub fn upload(host_bytes: []const u8) !DeviceBuffer {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        const dev_ptr = try CudaCachingAllocator.alloc(host_bytes.len);
        const rc = c.hk_cuda_upload(dev_ptr, host_bytes.ptr, host_bytes.len);
        if (rc != 0) {
            CudaCachingAllocator.free(dev_ptr, host_bytes.len);
            return CudaError.CudaCallFailed;
        }
        return DeviceBuffer{ .ptr = dev_ptr, .len = host_bytes.len };
    }

    pub fn allocUninit(nbytes: usize) !DeviceBuffer {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        const dev_ptr = try CudaCachingAllocator.alloc(nbytes);
        return DeviceBuffer{ .ptr = dev_ptr, .len = nbytes };
    }

    pub fn copyFromHost(self: DeviceBuffer, host_bytes: []const u8) !void {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        const copy_len = @min(self.len, host_bytes.len);
        const rc = c.hk_cuda_upload(self.ptr, host_bytes.ptr, copy_len);
        if (rc != 0) return CudaError.CudaCallFailed;
    }

    pub fn download(self: DeviceBuffer, host_out: []u8) !void {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        const rc = c.hk_cuda_download(host_out.ptr, self.ptr, @min(self.len, host_out.len));
        if (rc != 0) return CudaError.CudaCallFailed;
    }

    pub fn free(self: DeviceBuffer) void {
        if (!enabled) return;
        CudaCachingAllocator.free(self.ptr, self.len);
    }
};

pub fn synchronize() !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    if (c.hk_cuda_synchronize() != 0) return CudaError.CudaCallFailed;
}

pub fn uploadTo(dev_ptr: *anyopaque, host_bytes: []const u8) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_upload(dev_ptr, host_bytes.ptr, host_bytes.len);
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn downloadFrom(host_bytes: []u8, dev_ptr: *const anyopaque) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_download(host_bytes.ptr, dev_ptr, host_bytes.len);
    if (rc != 0) return CudaError.CudaCallFailed;
}

// ---------------------------------------------------------------------
// Unified Pointer Coercion Helpers
// ---------------------------------------------------------------------
inline fn toRawPtr(val: anytype) ?*anyopaque {
    const T = @TypeOf(val);
    if (T == DeviceBuffer) return val.ptr;
    if (T == ?DeviceBuffer) return if (val) |b| b.ptr else null;
    if (T == *anyopaque or T == ?*anyopaque) return val;
    if (T == *const anyopaque or T == ?*const anyopaque) return @constCast(val);
    return @ptrCast(val);
}

inline fn toConstRawPtr(val: anytype) ?*const anyopaque {
    const T = @TypeOf(val);
    if (T == DeviceBuffer) return val.ptr;
    if (T == ?DeviceBuffer) return if (val) |b| b.ptr else null;
    if (T == *const anyopaque or T == ?*const anyopaque) return val;
    if (T == *anyopaque or T == ?*anyopaque) return val;
    return @ptrCast(val);
}

// ---------------------------------------------------------------------
// Kernel Dispatches (Accepts DeviceBuffer or Raw Device Pointers)
// ---------------------------------------------------------------------
pub fn gemvF32(w: anytype, x: anytype, y: anytype, m: usize, k: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_gemv_f32(toConstRawPtr(w), toConstRawPtr(x), toRawPtr(y), @intCast(m), @intCast(k));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn gemvQ8_0(w: anytype, x: anytype, y: anytype, m: usize, k: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_gemv_q8_0(toConstRawPtr(w), toConstRawPtr(x), toRawPtr(y), @intCast(m), @intCast(k));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn gemvQ4_0(w: anytype, x: anytype, y: anytype, m: usize, k: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_gemv_q4_0(toConstRawPtr(w), toConstRawPtr(x), toRawPtr(y), @intCast(m), @intCast(k));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn matVec(w: anytype, st: format.StorageType, x: anytype, y: anytype, m: usize, k: usize) !void {
    switch (st) {
        .f32 => try gemvF32(w, x, y, m, k),
        .q8_0 => try gemvQ8_0(w, x, y, m, k),
        .q4_0 => try gemvQ4_0(w, x, y, m, k),
        else => return CudaError.UnsupportedStorageType,
    }
}

pub fn rmsNorm(x: anytype, weight: anytype, out: anytype, dim: usize, eps: f32) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_rmsnorm(toConstRawPtr(x), toConstRawPtr(weight), toRawPtr(out), @intCast(dim), eps);
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn headRmsNorm(x: anytype, weight: anytype, n_heads: usize, head_dim: usize, weight_len: usize, eps: f32) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_head_rmsnorm(toRawPtr(x), toConstRawPtr(weight), @intCast(n_heads), @intCast(head_dim), @intCast(weight_len), eps);
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn rope(q: anytype, k: anytype, pos: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, rope_theta: f32) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_rope(toRawPtr(q), toRawPtr(k), @intCast(pos), @intCast(n_heads), @intCast(n_kv_heads), @intCast(head_dim), rope_theta);
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn kvCacheUpdate(key_cache: anytype, val_cache: anytype, k: anytype, v: anytype, layer: usize, pos: usize, max_seq_len: usize, kv_dim: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_kv_cache_update(toRawPtr(key_cache), toRawPtr(val_cache), toConstRawPtr(k), toConstRawPtr(v), @intCast(layer), @intCast(pos), @intCast(max_seq_len), @intCast(kv_dim));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn gqaAttention(q: anytype, key_cache: anytype, val_cache: anytype, out: anytype, layer: usize, pos: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize, kv_dim: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_gqa_attention(toConstRawPtr(q), toConstRawPtr(key_cache), toConstRawPtr(val_cache), toRawPtr(out), @intCast(layer), @intCast(pos), @intCast(n_heads), @intCast(n_kv_heads), @intCast(head_dim), @intCast(max_seq_len), @intCast(kv_dim));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn swiglu(gate: anytype, up: anytype, hidden_dim: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_swiglu(toRawPtr(gate), toConstRawPtr(up), @intCast(hidden_dim));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn addResidual(x: anytype, res: anytype, dim: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_add_residual(toRawPtr(x), toConstRawPtr(res), @intCast(dim));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn fusedQkNormRope(q: anytype, k: anytype, wq: anytype, wk: anytype, pos: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, w_len_q: usize, w_len_k: usize, eps: f32, rope_theta: f32) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_fused_qknorm_rope(toRawPtr(q), toRawPtr(k), toConstRawPtr(wq), toConstRawPtr(wk), @intCast(pos), @intCast(n_heads), @intCast(n_kv_heads), @intCast(head_dim), @intCast(w_len_q), @intCast(w_len_k), eps, rope_theta);
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn fusedSwiGluResidual(gate: anytype, up: anytype, res: anytype, hidden_dim: usize, dim: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_fused_swiglu_residual(toRawPtr(gate), toConstRawPtr(up), toConstRawPtr(res), @intCast(hidden_dim), @intCast(dim));
    if (rc != 0) return CudaError.CudaCallFailed;
}

