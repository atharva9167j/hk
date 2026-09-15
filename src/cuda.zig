//! Zig bindings for the CUDA GPU backend (src/cuda/hk_cuda.cu).
//!
//! Only linked in when built with `-Dcuda=true`; otherwise every function
//! here returns error.CudaNotCompiledIn so callers can branch on
//! `cuda.enabled` at comptime or just try the call and handle the error.

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

/// Owning handle to a device allocation. Call `free()` when done.
pub const DeviceBuffer = struct {
    ptr: *anyopaque,
    len: usize,

    pub fn upload(host_bytes: []const u8) !DeviceBuffer {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        const dev_ptr = c.hk_cuda_malloc(host_bytes.len) orelse return CudaError.CudaCallFailed;
        const rc = c.hk_cuda_upload(dev_ptr, host_bytes.ptr, host_bytes.len);
        if (rc != 0) {
            c.hk_cuda_free(dev_ptr);
            return CudaError.CudaCallFailed;
        }
        return DeviceBuffer{ .ptr = dev_ptr, .len = host_bytes.len };
    }

    pub fn allocUninit(nbytes: usize) !DeviceBuffer {
        if (!enabled) return CudaError.CudaNotCompiledIn;
        const dev_ptr = c.hk_cuda_malloc(nbytes) orelse return CudaError.CudaCallFailed;
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
        c.hk_cuda_free(self.ptr);
    }
};

pub fn synchronize() !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    if (c.hk_cuda_synchronize() != 0) return CudaError.CudaCallFailed;
}

pub fn gemvF32(w: DeviceBuffer, x: DeviceBuffer, y: DeviceBuffer, m: usize, k: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_gemv_f32(w.ptr, x.ptr, y.ptr, @intCast(m), @intCast(k));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn gemvQ8_0(w: DeviceBuffer, x: DeviceBuffer, y: DeviceBuffer, m: usize, k: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_gemv_q8_0(w.ptr, x.ptr, y.ptr, @intCast(m), @intCast(k));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn gemvQ4_0(w: DeviceBuffer, x: DeviceBuffer, y: DeviceBuffer, m: usize, k: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_gemv_q4_0(w.ptr, x.ptr, y.ptr, @intCast(m), @intCast(k));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn matVec(w: DeviceBuffer, st: format.StorageType, x: DeviceBuffer, y: DeviceBuffer, m: usize, k: usize) !void {
    switch (st) {
        .f32 => try gemvF32(w, x, y, m, k),
        .q8_0 => try gemvQ8_0(w, x, y, m, k),
        .q4_0 => try gemvQ4_0(w, x, y, m, k),
        else => return CudaError.UnsupportedStorageType,
    }
}

pub fn rmsNorm(x: DeviceBuffer, weight: DeviceBuffer, out: DeviceBuffer, dim: usize, eps: f32) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_rmsnorm(x.ptr, weight.ptr, out.ptr, @intCast(dim), eps);
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn headRmsNorm(x: DeviceBuffer, weight: DeviceBuffer, n_heads: usize, head_dim: usize, weight_len: usize, eps: f32) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_head_rmsnorm(x.ptr, weight.ptr, @intCast(n_heads), @intCast(head_dim), @intCast(weight_len), eps);
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn rope(q: DeviceBuffer, k: DeviceBuffer, pos: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, rope_theta: f32) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_rope(q.ptr, k.ptr, @intCast(pos), @intCast(n_heads), @intCast(n_kv_heads), @intCast(head_dim), rope_theta);
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn kvCacheUpdate(key_cache: DeviceBuffer, val_cache: DeviceBuffer, k: DeviceBuffer, v: DeviceBuffer, layer: usize, pos: usize, max_seq_len: usize, kv_dim: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_kv_cache_update(key_cache.ptr, val_cache.ptr, k.ptr, v.ptr, @intCast(layer), @intCast(pos), @intCast(max_seq_len), @intCast(kv_dim));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn gqaAttention(q: DeviceBuffer, key_cache: DeviceBuffer, val_cache: DeviceBuffer, out: DeviceBuffer, layer: usize, pos: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize, kv_dim: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_gqa_attention(q.ptr, key_cache.ptr, val_cache.ptr, out.ptr, @intCast(layer), @intCast(pos), @intCast(n_heads), @intCast(n_kv_heads), @intCast(head_dim), @intCast(max_seq_len), @intCast(kv_dim));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn swiglu(gate: DeviceBuffer, up: DeviceBuffer, hidden_dim: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_swiglu(gate.ptr, up.ptr, @intCast(hidden_dim));
    if (rc != 0) return CudaError.CudaCallFailed;
}

pub fn addResidual(x: DeviceBuffer, res: DeviceBuffer, dim: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_add_residual(x.ptr, res.ptr, @intCast(dim));
    if (rc != 0) return CudaError.CudaCallFailed;
}
