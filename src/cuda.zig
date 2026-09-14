//! Zig bindings for the CUDA GPU backend (src/cuda/hk_cuda.cu).
//!
//! Only linked in when built with `-Dcuda=true`; otherwise every function
//! here returns error.CudaNotCompiledIn so callers can branch on
//! `cuda.enabled` at comptime or just try the call and handle the error.

const build_options = @import("build_options");

pub const enabled = build_options.cuda;

pub const CudaError = error{
    CudaNotCompiledIn,
    CudaCallFailed,
    NoCudaDevice,
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

/// y = W * x on-device. W is a device buffer holding an [M, K] row-major f32
/// matrix; x and y are device buffers of length K and M respectively.
pub fn gemvF32(w: DeviceBuffer, x: DeviceBuffer, y: DeviceBuffer, m: usize, k: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_gemv_f32(w.ptr, x.ptr, y.ptr, @intCast(m), @intCast(k));
    if (rc != 0) return CudaError.CudaCallFailed;
}

/// y = W * x on-device, W stored as Q8_0 quantized bytes (34 bytes / 32-elem
/// block: 2-byte f16 scale + 32 int8 values), matching tensor_ops.zig's
/// BlockQ8_0 layout exactly.
pub fn gemvQ8_0(w: DeviceBuffer, x: DeviceBuffer, y: DeviceBuffer, m: usize, k: usize) !void {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    const rc = c.hk_cuda_gemv_q8_0(w.ptr, x.ptr, y.ptr, @intCast(m), @intCast(k));
    if (rc != 0) return CudaError.CudaCallFailed;
}
