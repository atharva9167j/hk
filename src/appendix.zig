const std = @import("std");
const format = @import("format.zig");
const platform = @import("platform.zig");
const FileHeader = format.FileHeader;
const AppendixRecordHeader = format.AppendixRecordHeader;
const AppendixRecord = format.AppendixRecord;
const AppendixEntryType = format.AppendixEntryType;
const AppendixMetrics = format.AppendixMetrics;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const AppendixError = error{
    InvalidMagic,
    InvalidAppendixOffset,
    InvalidRecordHeader,
    TruncatedRecord,
    LineageMismatch,
    GenerationNotFound,
    FileNotFound,
    IoError,
    OutOfMemory,
};

pub const AppendixReader = struct {
    data: []const u8,
    records: std.ArrayList(AppendixRecord) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, file_data: []const u8) !AppendixReader {
        if (file_data.len < @sizeOf(FileHeader)) return AppendixError.InvalidMagic;
        const header: *const FileHeader = @ptrCast(@alignCast(file_data.ptr));
        if (!header.isValid()) return AppendixError.InvalidMagic;

        var reader = AppendixReader{
            .data = file_data,
            .records = .empty,
            .allocator = allocator,
        };

        if (header.appendix_offset > 0 and header.appendix_offset < file_data.len) {
            try reader.parseRecords(header.appendix_offset);
        }

        return reader;
    }

    pub fn deinit(self: *AppendixReader) void {
        self.records.deinit(self.allocator);
    }

    fn parseRecords(self: *AppendixReader, start_offset: u64) !void {
        var offset: usize = @intCast(start_offset);
        while (offset + @sizeOf(AppendixRecordHeader) <= self.data.len) {
            const rec_hdr_ptr: *const AppendixRecordHeader = @ptrCast(@alignCast(self.data.ptr + offset));
            const rec_hdr = rec_hdr_ptr.*;
            offset += @sizeOf(AppendixRecordHeader);

            const name_end = offset + rec_hdr.name_len;
            if (name_end > self.data.len) return AppendixError.TruncatedRecord;
            const name = self.data[offset..name_end];
            offset = name_end;

            const target_end = offset + rec_hdr.target_len;
            if (target_end > self.data.len) return AppendixError.TruncatedRecord;
            const target = self.data[offset..target_end];
            offset = target_end;

            const data_end = offset + @as(usize, @intCast(rec_hdr.data_size));
            if (data_end > self.data.len) return AppendixError.TruncatedRecord;
            const data_payload = self.data[offset..data_end];
            offset = data_end;

            // Align offset to 8 bytes for next record
            offset = std.mem.alignForward(usize, offset, 8);

            const entry_type: AppendixEntryType = @enumFromInt(rec_hdr.entry_type);

            try self.records.append(self.allocator, .{
                .entry_type = entry_type,
                .flags = rec_hdr.flags,
                .name = name,
                .target = target,
                .generation = rec_hdr.generation,
                .timestamp = rec_hdr.timestamp,
                .parent_hash = rec_hdr.parent_hash,
                .metrics = .{
                    .loss = rec_hdr.metric_loss,
                    .accuracy = rec_hdr.metric_acc,
                    .pass_rate = rec_hdr.metric_pass,
                    .custom = rec_hdr.metric_custom,
                },
                .data = data_payload,
            });
        }
    }

    pub fn verifyLineage(self: *const AppendixReader) bool {
        if (self.records.items.len <= 1) return true;
        for (1..self.records.items.len) |i| {
            const prev = self.records.items[i - 1];
            const curr = self.records.items[i];
            var hasher = Sha256.init(.{});
            hasher.update(prev.data);
            var expected_hash: [32]u8 = undefined;
            hasher.final(&expected_hash);

            var all_zero = true;
            for (curr.parent_hash) |b| {
                if (b != 0) {
                    all_zero = false;
                    break;
                }
            }
            if (!all_zero and !std.mem.eql(u8, &curr.parent_hash, &expected_hash)) {
                return false;
            }
        }
        return true;
    }
};

/// Append a new record directly to an existing HK file on disk
pub fn appendRecordToFile(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    record: AppendixRecord,
) !void {
    var region = try platform.mapOrReadFile(file_path, allocator);
    defer region.deinit(allocator);

    if (region.bytes.len < @sizeOf(FileHeader)) return AppendixError.InvalidMagic;
    const old_header: *const FileHeader = @ptrCast(@alignCast(region.bytes.ptr));
    if (!old_header.isValid()) return AppendixError.InvalidMagic;

    const file_size = region.bytes.len;
    const append_offset = std.mem.alignForward(usize, file_size, 8);
    const rec_body_len = @sizeOf(AppendixRecordHeader) + record.name.len + record.target.len + record.data.len;
    const aligned_rec_len = std.mem.alignForward(usize, rec_body_len, 8);
    const new_file_size = append_offset + aligned_rec_len;

    const new_buf = try allocator.alloc(u8, new_file_size);
    defer allocator.free(new_buf);

    // Copy existing file content
    @memcpy(new_buf[0..file_size], region.bytes);
    // Zero out any alignment gap between old file_size and append_offset
    if (append_offset > file_size) {
        @memset(new_buf[file_size..append_offset], 0);
    }

    // Update Header in new buffer
    var new_hdr = old_header.*;
    if (new_hdr.appendix_offset == 0) {
        new_hdr.appendix_offset = @intCast(append_offset);
        new_hdr.flags |= format.HeaderFlags.HAS_APPENDIX;
    }
    const hdr_bytes = std.mem.asBytes(&new_hdr);
    @memcpy(new_buf[0..@sizeOf(FileHeader)], hdr_bytes);

    // Write AppendixRecordHeader
    const rec_hdr = AppendixRecordHeader{
        .entry_type = @intFromEnum(record.entry_type),
        .flags = record.flags,
        .name_len = @intCast(record.name.len),
        .generation = record.generation,
        .timestamp = record.timestamp,
        .parent_hash = record.parent_hash,
        .metric_loss = record.metrics.loss,
        .metric_acc = record.metrics.accuracy,
        .metric_pass = record.metrics.pass_rate,
        .metric_custom = record.metrics.custom,
        .target_len = @intCast(record.target.len),
        .reserved = 0,
        .data_crc32 = 0,
        .data_size = record.data.len,
    };
    const rec_hdr_bytes = std.mem.asBytes(&rec_hdr);

    var cur = append_offset;
    @memcpy(new_buf[cur .. cur + @sizeOf(AppendixRecordHeader)], rec_hdr_bytes);
    cur += @sizeOf(AppendixRecordHeader);

    @memcpy(new_buf[cur .. cur + record.name.len], record.name);
    cur += record.name.len;

    @memcpy(new_buf[cur .. cur + record.target.len], record.target);
    cur += record.target.len;

    @memcpy(new_buf[cur .. cur + record.data.len], record.data);
    cur += record.data.len;

    // Pad trailing bytes to 8-byte boundary
    if (cur < new_file_size) {
        @memset(new_buf[cur..new_file_size], 0);
    }

    // Write back atomically
    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();
    var out_file = try cwd.createFile(io, file_path, .{});
    defer out_file.close(io);
    try out_file.writeStreamingAll(io, new_buf);
}

/// Rollback an HK file by truncating appendix records beyond target_generation
pub fn rollbackToFile(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    target_generation: u32,
) !void {
    var region = try platform.mapOrReadFile(file_path, allocator);
    defer region.deinit(allocator);

    if (region.bytes.len < @sizeOf(FileHeader)) return AppendixError.InvalidMagic;
    const header: *const FileHeader = @ptrCast(@alignCast(region.bytes.ptr));
    if (!header.isValid()) return AppendixError.InvalidMagic;
    if (header.appendix_offset == 0 or header.appendix_offset >= region.bytes.len) return;

    var offset: usize = @intCast(header.appendix_offset);
    var truncate_pos: usize = offset;
    var found = false;

    while (offset + @sizeOf(AppendixRecordHeader) <= region.bytes.len) {
        const rec_hdr_ptr: *const AppendixRecordHeader = @ptrCast(@alignCast(region.bytes.ptr + offset));
        const rec_hdr = rec_hdr_ptr.*;

        offset += @sizeOf(AppendixRecordHeader);
        offset += rec_hdr.name_len;
        offset += rec_hdr.target_len;
        offset += @intCast(rec_hdr.data_size);
        offset = std.mem.alignForward(usize, offset, 8);

        if (rec_hdr.generation <= target_generation) {
            truncate_pos = offset;
            found = true;
        } else {
            break;
        }
    }

    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();

    if (!found and target_generation == 0) {
        // Rollback all appendix entries: reset appendix_offset to 0
        var new_buf = try allocator.alloc(u8, @intCast(header.appendix_offset));
        defer allocator.free(new_buf);
        @memcpy(new_buf, region.bytes[0..@intCast(header.appendix_offset)]);

        var hdr_copy = header.*;
        hdr_copy.appendix_offset = 0;
        hdr_copy.flags &= ~format.HeaderFlags.HAS_APPENDIX;
        const hdr_bytes = std.mem.asBytes(&hdr_copy);
        @memcpy(new_buf[0..@sizeOf(FileHeader)], hdr_bytes);

        var out_file = try cwd.createFile(io, file_path, .{});
        defer out_file.close(io);
        try out_file.writeStreamingAll(io, new_buf);
    } else {
        const out_slice = region.bytes[0..truncate_pos];
        var out_file = try cwd.createFile(io, file_path, .{});
        defer out_file.close(io);
        try out_file.writeStreamingAll(io, out_slice);
    }
}
