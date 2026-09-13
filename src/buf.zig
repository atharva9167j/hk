const std = @import("std");

pub const BufferWriter = struct {
    list: std.ArrayList(u8) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BufferWriter {
        return .{
            .list = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BufferWriter) void {
        self.list.deinit(self.allocator);
    }

    pub fn writeU8(self: *BufferWriter, val: u8) !void {
        try self.list.append(self.allocator, val);
    }

    pub fn writeU16(self: *BufferWriter, val: u16) !void {
        var b: [2]u8 = undefined;
        std.mem.writeInt(u16, &b, val, .little);
        try self.list.appendSlice(self.allocator, &b);
    }

    pub fn writeU32(self: *BufferWriter, val: u32) !void {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, val, .little);
        try self.list.appendSlice(self.allocator, &b);
    }

    pub fn writeU64(self: *BufferWriter, val: u64) !void {
        var b: [8]u8 = undefined;
        std.mem.writeInt(u64, &b, val, .little);
        try self.list.appendSlice(self.allocator, &b);
    }

    pub fn writeI64(self: *BufferWriter, val: i64) !void {
        var b: [8]u8 = undefined;
        std.mem.writeInt(i64, &b, val, .little);
        try self.list.appendSlice(self.allocator, &b);
    }

    pub fn writeF64(self: *BufferWriter, val: f64) !void {
        const bits: u64 = @bitCast(val);
        try self.writeU64(bits);
    }

    pub fn writeBytes(self: *BufferWriter, bytes: []const u8) !void {
        try self.list.appendSlice(self.allocator, bytes);
    }

    pub fn getBytes(self: *const BufferWriter) []const u8 {
        return self.list.items;
    }
};

pub const BufferReader = struct {
    buffer: []const u8,
    pos: usize = 0,

    pub fn init(buffer: []const u8) BufferReader {
        return .{ .buffer = buffer, .pos = 0 };
    }

    pub fn readU8(self: *BufferReader) !u8 {
        if (self.pos + 1 > self.buffer.len) return error.UnexpectedEof;
        const val = self.buffer[self.pos];
        self.pos += 1;
        return val;
    }

    pub fn readU16(self: *BufferReader) !u16 {
        if (self.pos + 2 > self.buffer.len) return error.UnexpectedEof;
        const val = std.mem.readInt(u16, self.buffer[self.pos..][0..2], .little);
        self.pos += 2;
        return val;
    }

    pub fn readU32(self: *BufferReader) !u32 {
        if (self.pos + 4 > self.buffer.len) return error.UnexpectedEof;
        const val = std.mem.readInt(u32, self.buffer[self.pos..][0..4], .little);
        self.pos += 4;
        return val;
    }

    pub fn readU64(self: *BufferReader) !u64 {
        if (self.pos + 8 > self.buffer.len) return error.UnexpectedEof;
        const val = std.mem.readInt(u64, self.buffer[self.pos..][0..8], .little);
        self.pos += 8;
        return val;
    }

    pub fn readI64(self: *BufferReader) !i64 {
        if (self.pos + 8 > self.buffer.len) return error.UnexpectedEof;
        const val = std.mem.readInt(i64, self.buffer[self.pos..][0..8], .little);
        self.pos += 8;
        return val;
    }

    pub fn readF64(self: *BufferReader) !f64 {
        const bits = try self.readU64();
        return @bitCast(bits);
    }

    pub fn readBytes(self: *BufferReader, len: usize) ![]const u8 {
        if (self.pos + len > self.buffer.len) return error.UnexpectedEof;
        const s = self.buffer[self.pos .. self.pos + len];
        self.pos += len;
        return s;
    }
};
