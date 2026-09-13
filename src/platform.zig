const std = @import("std");

pub const PAGE_SIZE: usize = 4096;

pub const MmapRegion = struct {
    bytes: []align(PAGE_SIZE) u8,

    pub fn deinit(self: *MmapRegion, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
    }
};

/// Aligns an offset forward to the next multiple of alignment
pub fn alignForward(offset: usize, alignment: usize) usize {
    const rem = offset % alignment;
    if (rem == 0) return offset;
    return offset + (alignment - rem);
}

pub fn isAligned(offset: usize, alignment: usize) bool {
    return (offset % alignment) == 0;
}

/// Reads or maps an entire file into memory with page alignment
pub fn mapOrReadFile(path: []const u8, allocator: std.mem.Allocator) !MmapRegion {
    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();

    var file = try cwd.openFile(io, path, .{});
    defer file.close(io);

    const st = try file.stat(io);
    const size: usize = @intCast(st.size);

    const aligned_slice = try allocator.alignedAlloc(u8, .fromByteUnits(PAGE_SIZE), size);
    errdefer allocator.free(aligned_slice);

    var slices = [_][]u8{aligned_slice};
    const read_len = try file.readStreaming(io, &slices);
    if (read_len != size) {
        return error.UnexpectedEof;
    }

    return MmapRegion{
        .bytes = aligned_slice,
    };
}

/// Writes bytes into an existing file at a specific offset without rewriting other bytes
pub fn writeBytesAtOffset(path: []const u8, data: []const u8, offset: u64, allocator: std.mem.Allocator) !void {
    if (data.len == 0) return;
    if (@import("builtin").os.tag == .windows) {
        const win = struct {
            extern "kernel32" fn CreateFileA(
                lpFileName: [*:0]const u8,
                dwDesiredAccess: u32,
                dwShareMode: u32,
                lpSecurityAttributes: ?*anyopaque,
                dwCreationDisposition: u32,
                dwFlagsAndAttributes: u32,
                hTemplateFile: ?*anyopaque,
            ) callconv(.winapi) std.os.windows.HANDLE;

            extern "kernel32" fn WriteFile(
                hFile: std.os.windows.HANDLE,
                lpBuffer: [*]const u8,
                nNumberOfBytesToWrite: u32,
                lpNumberOfBytesWritten: ?*u32,
                lpOverlapped: ?*anyopaque,
            ) callconv(.winapi) i32;

            extern "kernel32" fn SetFilePointer(
                hFile: std.os.windows.HANDLE,
                lDistanceToMove: i32,
                lpDistanceToMoveHigh: ?*i32,
                dwMoveMethod: u32,
            ) callconv(.winapi) u32;

            extern "kernel32" fn CloseHandle(hObject: std.os.windows.HANDLE) callconv(.winapi) i32;
        };

        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);

        const handle = win.CreateFileA(path_z.ptr, 0x40000000 | 0x80000000, 1 | 2, null, 3, 0x80, null);
        if (handle == std.os.windows.INVALID_HANDLE_VALUE) return error.FileNotFound;
        defer _ = win.CloseHandle(handle);

        var high: i32 = @intCast((offset >> 32) & 0xFFFFFFFF);
        const low: i32 = @intCast(offset & 0xFFFFFFFF);
        _ = win.SetFilePointer(handle, low, &high, 0);

        var total_written: usize = 0;
        while (total_written < data.len) {
            var written: u32 = 0;
            const chunk_len: u32 = @intCast(@min(data.len - total_written, 1024 * 1024 * 64));
            const ok = win.WriteFile(handle, data[total_written..].ptr, chunk_len, &written, null);
            if (ok == 0) return error.WriteFailed;
            total_written += written;
        }
    } else {
        const io = std.Options.debug_io;
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.openFile(io, path, .{ .mode = .read_write });
        defer file.close(io);
        try file.writePositionalAll(io, data, offset);
    }
}

