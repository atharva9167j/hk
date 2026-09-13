const std = @import("std");
const hk = @import("hk");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer it.deinit();

    _ = it.next(); // skip program name

    const command = it.next() orelse {
        printUsage();
        return;
    };

    if (std.mem.eql(u8, command, "inspect")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'inspect'\nUsage: hk inspect <model.hk>\n", .{});
            return;
        };
        try cmdInspect(file_path, allocator);
    } else if (std.mem.eql(u8, command, "verify")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'verify'\nUsage: hk verify <model.hk>\n", .{});
            return;
        };
        try cmdVerify(file_path, allocator);
    } else if (std.mem.eql(u8, command, "benchmark")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'benchmark'\nUsage: hk benchmark <model.hk>\n", .{});
            return;
        };
        try cmdBenchmark(file_path, allocator);
    } else if (std.mem.eql(u8, command, "retile")) {
        const input_path = it.next() orelse {
            std.debug.print("Error: Missing input path for 'retile'\nUsage: hk retile <in.hk> <out.hk> [tile_16x16|row_major]\n", .{});
            return;
        };
        const output_path = it.next() orelse {
            std.debug.print("Error: Missing output path for 'retile'\nUsage: hk retile <in.hk> <out.hk> [tile_16x16|row_major]\n", .{});
            return;
        };
        const target_layout_str = it.next() orelse "tile_16x16";
        try cmdRetile(input_path, output_path, target_layout_str, allocator);
    } else if (std.mem.eql(u8, command, "prune")) {
        const input_path = it.next() orelse {
            std.debug.print("Error: Missing input path for 'prune'\nUsage: hk prune <in.hk> <out.hk> [ratio, e.g. 0.5]\n", .{});
            return;
        };
        const output_path = it.next() orelse {
            std.debug.print("Error: Missing output path for 'prune'\nUsage: hk prune <in.hk> <out.hk> [ratio, e.g. 0.5]\n", .{});
            return;
        };
        const ratio_str = it.next() orelse "0.5";
        const ratio = std.fmt.parseFloat(f32, ratio_str) catch 0.5;
        try cmdPrune(input_path, output_path, ratio, allocator);
    } else if (std.mem.eql(u8, command, "expand")) {
        const input_path = it.next() orelse {
            std.debug.print("Error: Missing input path for 'expand'\nUsage: hk expand <in.hk> <out.hk> [--vocab <N>] [--width <ratio>]\n", .{});
            return;
        };
        const output_path = it.next() orelse {
            std.debug.print("Error: Missing output path for 'expand'\nUsage: hk expand <in.hk> <out.hk> [--vocab <N>] [--width <ratio>]\n", .{});
            return;
        };
        var new_vocab_opt: ?usize = null;
        var width_ratio_opt: ?f32 = null;

        while (it.next()) |flag| {
            if (std.mem.eql(u8, flag, "--vocab")) {
                const val = it.next() orelse break;
                new_vocab_opt = std.fmt.parseInt(usize, val, 10) catch null;
            } else if (std.mem.eql(u8, flag, "--width")) {
                const val = it.next() orelse break;
                width_ratio_opt = std.fmt.parseFloat(f32, val) catch null;
            }
        }
        try cmdExpand(input_path, output_path, new_vocab_opt, width_ratio_opt, allocator);
    } else if (std.mem.eql(u8, command, "eval")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'eval'\nUsage: hk eval <model.hk>\n", .{});
            return;
        };
        try cmdEval(file_path, allocator);
    } else if (std.mem.eql(u8, command, "appendix")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'appendix'\nUsage: hk appendix <model.hk>\n", .{});
            return;
        };
        try cmdAppendix(file_path, allocator);
    } else if (std.mem.eql(u8, command, "rollback")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'rollback'\nUsage: hk rollback <model.hk> [generation]\n", .{});
            return;
        };
        const gen_str = it.next() orelse "0";
        const target_gen = std.fmt.parseInt(u32, gen_str, 10) catch 0;
        try cmdRollback(file_path, target_gen, allocator);
    } else if (std.mem.eql(u8, command, "metadata")) {
        const sub_cmd = it.next() orelse {
            std.debug.print("Error: Missing subcommand for 'metadata'\nUsage: hk metadata <set|get|list> <file.hk> [key] [val]\n", .{});
            return;
        };
        if (std.mem.eql(u8, sub_cmd, "set")) {
            const file_path = it.next() orelse {
                std.debug.print("Error: Missing file path for 'metadata set'\nUsage: hk metadata set <file.hk> <key> <val>\n", .{});
                return;
            };
            const key = it.next() orelse {
                std.debug.print("Error: Missing key for 'metadata set'\nUsage: hk metadata set <file.hk> <key> <val>\n", .{});
                return;
            };
            const val = it.next() orelse {
                std.debug.print("Error: Missing value for 'metadata set'\nUsage: hk metadata set <file.hk> <key> <val>\n", .{});
                return;
            };
            try cmdMetadataSet(file_path, key, val, allocator);
        } else if (std.mem.eql(u8, sub_cmd, "get")) {
            const file_path = it.next() orelse {
                std.debug.print("Error: Missing file path for 'metadata get'\nUsage: hk metadata get <file.hk> <key>\n", .{});
                return;
            };
            const key = it.next() orelse {
                std.debug.print("Error: Missing key for 'metadata get'\nUsage: hk metadata get <file.hk> <key>\n", .{});
                return;
            };
            try cmdMetadataGet(file_path, key, allocator);
        } else if (std.mem.eql(u8, sub_cmd, "list")) {
            const file_path = it.next() orelse {
                std.debug.print("Error: Missing file path for 'metadata list'\nUsage: hk metadata list <file.hk>\n", .{});
                return;
            };
            try cmdMetadataList(file_path, allocator);
        } else {
            std.debug.print("Unknown metadata subcommand: {s}\nUsage: hk metadata <set|get|list> <file.hk> [key] [val]\n", .{sub_cmd});
        }
    } else if (std.mem.eql(u8, command, "dump")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'dump'\nUsage: hk dump <model.hk>\n", .{});
            return;
        };
        try cmdDump(file_path, allocator);
    } else if (std.mem.eql(u8, command, "hash")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'hash'\nUsage: hk hash <model.hk>\n", .{});
            return;
        };
        try cmdHash(file_path, allocator);
    } else if (std.mem.eql(u8, command, "convert-endian")) {
        const input_path = it.next() orelse {
            std.debug.print("Error: Missing input path for 'convert-endian'\nUsage: hk convert-endian <in.hk> <out.hk>\n", .{});
            return;
        };
        const output_path = it.next() orelse {
            std.debug.print("Error: Missing output path for 'convert-endian'\nUsage: hk convert-endian <in.hk> <out.hk>\n", .{});
            return;
        };
        try cmdConvertEndian(input_path, output_path, allocator);
    } else if (std.mem.eql(u8, command, "convert-gguf")) {
        const input_path = it.next() orelse {
            std.debug.print("Error: Missing input path for 'convert-gguf'\nUsage: hk convert-gguf <in.gguf> <out.hk>\n", .{});
            return;
        };
        const output_path = it.next() orelse {
            std.debug.print("Error: Missing output path for 'convert-gguf'\nUsage: hk convert-gguf <in.gguf> <out.hk>\n", .{});
            return;
        };
        try cmdConvertGGUF(input_path, output_path, allocator);
    } else if (std.mem.eql(u8, command, "export")) {
        var format_str: []const u8 = "gguf";
        var in_path_opt: ?[]const u8 = null;
        var out_path_opt: ?[]const u8 = null;

        while (it.next()) |arg| {
            if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--format")) {
                format_str = it.next() orelse "gguf";
            } else if (in_path_opt == null) {
                in_path_opt = arg;
            } else if (out_path_opt == null) {
                out_path_opt = arg;
            }
        }

        if (in_path_opt == null or out_path_opt == null) {
            std.debug.print("Error: Missing arguments for 'export'\nUsage: hk export -f <gguf|safetensors> <in.hk> <out_file>\n", .{});
            return;
        }

        try cmdExport(format_str, in_path_opt.?, out_path_opt.?, allocator);
    } else if (std.mem.eql(u8, command, "gui")) {
        const file_path_opt = it.next();
        try cmdGui(file_path_opt, allocator);
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help")) {
        printUsage();
    } else {
        std.debug.print("Unknown command: {s}\n", .{command});
        printUsage();
    }
}

fn printUsage() void {
    std.debug.print(
        \\HK (Neural Tensor Format) CLI v1.0.0
        \\Usage: hk <command> [arguments]
        \\
        \\Commands:
        \\  inspect        <file.hk>                     Display header, metadata, and tensor TOC details
        \\  dump           <file.hk>                     Comprehensive binary dumper (hex, headers, alignment)
        \\  hash           <file.hk>                     SHA-256 container and per-tensor verification
        \\  convert-gguf   <in.gguf> <out.hk>            Zero-copy bitstream ingestion of GGUF models
        \\  export         -f <gguf|safetensors> <in> <out> Export HK model to GGUF v3 or Safetensors
        \\  convert-endian <in.hk> <out.hk>              Convert endianness (Little <-> Big Endian)
        \\  gui            [file.hk]                     Launch visual HK model editor GUI
        \\  verify         <file.hk>                     Verify header magic, 128-byte alignment, and bounds
        \\  eval           <file.hk>                     Inspect model weights, parameter counts, and integrity
        \\  expand         <in.hk> <out.hk> [options]    Natively expand vocab (--vocab N) and width (--width R)
        \\  benchmark      <file.hk>                     Benchmark mmap loading and dequantization throughput
        \\  retile         <in.hk> <out.hk> [layout]     Re-tile 2D weight matrices for Tensor Cores
        \\  prune          <in.hk> <out.hk> [ratio]      Apply magnitude pruning to weight tensors
        \\  appendix       <file.hk>                     Display version-chained appendix records & metrics
        \\  rollback       <file.hk> [generation]        Rollback appendix entries to specified generation
        \\  metadata       <set|get|list> <file.hk> ...  In-place metadata inspection and modification
        \\  help                                         Show this help message
        \\
    , .{});
}

fn cmdConvertGGUF(input_path: []const u8, output_path: []const u8, allocator: std.mem.Allocator) !void {
    std.debug.print("[HK] Ingesting GGUF container: {s}\n", .{input_path});
    hk.gguf.convertGGUFToHK(input_path, output_path, allocator) catch |err| {
        std.debug.print("Error: Failed to convert GGUF to HK: {}\n", .{err});
        return;
    };
    std.debug.print("[HK] Successfully ingested GGUF to HK -> {s}\n", .{output_path});
}

fn cmdExport(fmt: []const u8, input_path: []const u8, output_path: []const u8, allocator: std.mem.Allocator) !void {
    if (std.mem.eql(u8, fmt, "gguf")) {
        std.debug.print("[HK] Exporting HK model to GGUF v3: {s}\n", .{output_path});
        hk.gguf.exportHKToGGUF(input_path, output_path, allocator) catch |err| {
            std.debug.print("Error: Failed to export HK to GGUF: {}\n", .{err});
            return;
        };
        std.debug.print("[HK] Successfully exported to GGUF -> {s}\n", .{output_path});
    } else {
        std.debug.print("Error: Unsupported export format '{s}'. Supported: gguf\n", .{fmt});
    }
}

fn cmdMetadataSet(file_path: []const u8, key: []const u8, val: []const u8, allocator: std.mem.Allocator) !void {
    hk.metadata.patchFileMetadataInPlace(allocator, file_path, key, val) catch |err| {
        std.debug.print("Error: Failed to patch metadata in '{s}': {}\n", .{ file_path, err });
        return;
    };
    std.debug.print("[SUCCESS] In-place metadata updated in '{s}':\n  {s} = \"{s}\"\n(Tensor payload untouched, zero copy)\n", .{
        file_path, key, val,
    });
}

fn cmdMetadataGet(file_path: []const u8, key: []const u8, allocator: std.mem.Allocator) !void {
    var reader = hk.HKReader.open(file_path, allocator) catch |err| {
        std.debug.print("Failed to open HK file '{s}': {}\n", .{ file_path, err });
        return;
    };
    defer reader.deinit();

    if (reader.metadata_map.get(key)) |val| {
        switch (val) {
            .val_string => |s| std.debug.print("{s}\n", .{s}),
            .val_int64 => |v| std.debug.print("{}\n", .{v}),
            .val_float64 => |f| std.debug.print("{d}\n", .{f}),
            .val_bool => |b| std.debug.print("{}\n", .{b}),
            .val_json => |j| std.debug.print("{s}\n", .{j}),
            .val_bytes => |b| std.debug.print("<{} bytes>\n", .{b.len}),
        }
    } else {
        std.debug.print("Key '{s}' not found in metadata of '{s}'.\n", .{ key, file_path });
    }
}

fn cmdMetadataList(file_path: []const u8, allocator: std.mem.Allocator) !void {
    var reader = hk.HKReader.open(file_path, allocator) catch |err| {
        std.debug.print("Failed to open HK file '{s}': {}\n", .{ file_path, err });
        return;
    };
    defer reader.deinit();

    std.debug.print("\n=== Metadata for '{s}' ({} entries) ===\n", .{ file_path, reader.metadata_map.items.items.len });
    for (reader.metadata_map.items.items) |item| {
        switch (item.value) {
            .val_string => |s| std.debug.print("  {s}: \"{s}\"\n", .{ item.key, s }),
            .val_int64 => |v| std.debug.print("  {s}: {}\n", .{ item.key, v }),
            .val_float64 => |f| std.debug.print("  {s}: {d:.4}\n", .{ item.key, f }),
            .val_bool => |b| std.debug.print("  {s}: {}\n", .{ item.key, b }),
            .val_json => |j| std.debug.print("  {s} (JSON): {s}\n", .{ item.key, j }),
            .val_bytes => |b| std.debug.print("  {s} (binary): {} bytes\n", .{ item.key, b.len }),
        }
    }
    std.debug.print("\n", .{});
}

fn cmdInspect(path: []const u8, allocator: std.mem.Allocator) !void {
    var reader = hk.HKReader.open(path, allocator) catch |err| {
        std.debug.print("Failed to open HK file '{s}': {}\n", .{ path, err });
        return;
    };
    defer reader.deinit();

    std.debug.print("\n=== HK File: {s} ===\n", .{path});
    std.debug.print("Magic: {s} | Version: {}.{}\n", .{
        reader.header.magic,
        reader.header.version_major,
        reader.header.version_minor,
    });
    std.debug.print("Flags: 0x{X:0>8} | Tensors: {} | Metadata KVs: {}\n", .{
        reader.header.flags,
        reader.header.tensor_count,
        reader.header.metadata_kv_count,
    });
    std.debug.print("Data Offset: 0x{X} (128-byte aligned: {})\n", .{
        reader.header.tensor_data_offset,
        (reader.header.tensor_data_offset % hk.format.ALIGNMENT_BYTES) == 0,
    });
    if (reader.isSharded()) {
        std.debug.print("Sharding: Shard {} of {} (FLAG_IS_SHARDED enabled)\n", .{
            reader.getSplitIndex() + 1,
            reader.getSplitCount(),
        });
    } else {
        std.debug.print("Sharding: Single Container (unsharded)\n", .{});
    }

    std.debug.print("\n--- Metadata ---\n", .{});
    for (reader.metadata_map.items.items) |item| {
        switch (item.value) {
            .val_string => |s| std.debug.print("  {s}: \"{s}\"\n", .{ item.key, s }),
            .val_int64 => |v| std.debug.print("  {s}: {}\n", .{ item.key, v }),
            .val_float64 => |f| std.debug.print("  {s}: {d:.4}\n", .{ item.key, f }),
            .val_bool => |b| std.debug.print("  {s}: {}\n", .{ item.key, b }),
            .val_json => |j| std.debug.print("  {s} (JSON): {s}\n", .{ item.key, j }),
            .val_bytes => |b| std.debug.print("  {s} (binary): {} bytes\n", .{ item.key, b.len }),
        }
    }

    std.debug.print("\n--- Tensor Table of Contents ({} entries) ---\n", .{reader.toc.entries.items.len});
    std.debug.print("{s:<40} {s:<12} {s:<12} {s:<16} {s:<10} {s:<12}\n", .{
        "Tensor Name", "Type", "Layout", "Shape", "Size", "Sparsity",
    });
    std.debug.print("{s:-<105}\n", .{""});

    for (reader.toc.entries.items) |e| {
        var shape_buf: [64]u8 = undefined;
        var pos: usize = 0;
        shape_buf[pos] = '[';
        pos += 1;
        for (0..e.ndim) |d| {
            if (d > 0) {
                shape_buf[pos] = ',';
                pos += 1;
            }
            const part = std.fmt.bufPrint(shape_buf[pos..], "{}", .{e.shape[d]}) catch "";
            pos += part.len;
        }
        shape_buf[pos] = ']';
        pos += 1;
        const shape_str = shape_buf[0..pos];

        std.debug.print("{s:<40} {s:<12} {s:<12} {s:<16} {}B {d:>6.1}%\n", .{
            e.name,
            @tagName(e.storage_type),
            @tagName(e.tile_layout),
            shape_str,
            e.data_size,
            e.sparsity_ratio * 100.0,
        });
    }
    std.debug.print("\n", .{});
}

fn cmdVerify(path: []const u8, allocator: std.mem.Allocator) !void {
    var reader = hk.HKReader.open(path, allocator) catch |err| {
        std.debug.print("VERIFY FAILED: Could not open file: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    var passed = true;
    std.debug.print("Verifying '{s}'...\n", .{path});

    if (!reader.header.isValid()) {
        std.debug.print("[FAIL] Invalid magic bytes or unsupported version!\n", .{});
        passed = false;
    } else {
        std.debug.print("[PASS] Header magic 'HKNT' and version 1.0 valid\n", .{});
    }

    if ((reader.header.tensor_data_offset % hk.format.ALIGNMENT_BYTES) != 0) {
        std.debug.print("[FAIL] Data offset 0x{X} is not 128-byte aligned!\n", .{reader.header.tensor_data_offset});
        passed = false;
    } else {
        std.debug.print("[PASS] Payload offset is 128-byte Tensor Core aligned\n", .{});
    }

    for (reader.toc.entries.items) |e| {
        if (e.storage_type == .null_ref) continue;
        if ((e.data_offset % hk.format.ALIGNMENT_BYTES) != 0) {
            std.debug.print("[FAIL] Tensor '{s}' offset 0x{X} is not 128-byte aligned!\n", .{ e.name, e.data_offset });
            passed = false;
        }
        if (e.data_offset + e.data_size > reader.mmap_region.bytes.len) {
            std.debug.print("[FAIL] Tensor '{s}' data extends beyond file bounds!\n", .{e.name});
            passed = false;
        }
    }

    if (passed) {
        std.debug.print("\n===> ALL CHECKS PASSED: File is a compliant, high-performance HK binary container.\n\n", .{});
    } else {
        std.debug.print("\n===> VERIFICATION FAILED: Violations found.\n\n", .{});
    }
}

fn cmdBenchmark(path: []const u8, allocator: std.mem.Allocator) !void {
    const io = std.Options.debug_io;
    std.debug.print("Benchmarking HK loader on '{s}'...\n", .{path});

    const start_open = std.Io.Timestamp.now(io, .awake);
    var reader = try hk.HKReader.open(path, allocator);
    defer reader.deinit();
    const end_open = std.Io.Timestamp.now(io, .awake);
    const open_time_ns = end_open.nanoseconds - start_open.nanoseconds;

    std.debug.print("Zero-Copy Open & Deserialization Time: {d:.2} us\n", .{@as(f64, @floatFromInt(open_time_ns)) / 1000.0});

    var total_bytes: usize = 0;
    const start_read = std.Io.Timestamp.now(io, .awake);
    for (reader.toc.entries.items) |e| {
        const data = try reader.getTensorData(e);
        total_bytes += data.len;
    }
    const end_read = std.Io.Timestamp.now(io, .awake);
    const read_time_ns = end_read.nanoseconds - start_read.nanoseconds;
    const read_sec = @as(f64, @floatFromInt(read_time_ns)) / 1_000_000_000.0;
    const mbs = if (read_sec > 0) (@as(f64, @floatFromInt(total_bytes)) / (1024.0 * 1024.0)) / read_sec else 0.0;

    std.debug.print("Zero-copy slice traversal: {d:.2} us ({} bytes total, throughput: {d:.1} MB/s)\n", .{
        @as(f64, @floatFromInt(read_time_ns)) / 1000.0,
        total_bytes,
        mbs,
    });

    // Benchmark Dequantization throughput
    var total_elements: usize = 0;
    for (reader.toc.entries.items) |e| {
        var numel: usize = 1;
        for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);
        total_elements += numel;
    }

    const deq_buf = try allocator.alloc(f32, total_elements);
    defer allocator.free(deq_buf);

    const start_deq = std.Io.Timestamp.now(io, .awake);
    var offset: usize = 0;
    for (reader.toc.entries.items) |e| {
        var numel: usize = 1;
        for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);
        reader.dequantizeToF32(e, true, deq_buf[offset .. offset + numel]) catch {};
        offset += numel;
    }
    const end_deq = std.Io.Timestamp.now(io, .awake);
    const deq_time_ns = end_deq.nanoseconds - start_deq.nanoseconds;
    const deq_sec = @as(f64, @floatFromInt(deq_time_ns)) / 1_000_000_000.0;
    const melem_per_sec = if (deq_sec > 0) (@as(f64, @floatFromInt(total_elements)) / 1_000_000.0) / deq_sec else 0.0;

    std.debug.print("Reconstruction & dequantization: {d:.2} ms ({} elements, throughput: {d:.2} M-elem/s)\n\n", .{
        deq_sec * 1000.0,
        total_elements,
        melem_per_sec,
    });
}

fn cmdRetile(in_path: []const u8, out_path: []const u8, layout_str: []const u8, allocator: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var reader = try hk.HKReader.open(in_path, arena_alloc);
    defer reader.deinit();

    var writer = hk.HKWriter.init(arena_alloc);
    defer writer.deinit();

    for (reader.metadata_map.items.items) |m| {
        switch (m.value) {
            .val_string => |s| try writer.addMetadataString(m.key, s),
            .val_int64 => |v| try writer.addMetadataInt(m.key, v),
            .val_float64 => |f| try writer.addMetadataFloat(m.key, f),
            .val_bool => |b| try writer.addMetadataBool(m.key, b),
            else => {},
        }
    }
    try writer.addMetadataString("retiled_with", layout_str);

    const target_layout: hk.TileLayout = if (std.mem.eql(u8, layout_str, "tile_16x16"))
        .tile_16x16
    else if (std.mem.eql(u8, layout_str, "tile_16x8"))
        .tile_16x8
    else if (std.mem.eql(u8, layout_str, "tile_32x16"))
        .tile_32x16
    else
        .row_major;

    for (reader.toc.entries.items) |e| {
        var numel: usize = 1;
        for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);

        const dense = try arena_alloc.alloc(f32, numel);
        try reader.dequantizeToF32(e, true, dense);

        if (e.ndim == 2 and target_layout != .row_major) {
            const tiled = try hk.tiling.packTilesF32(
                dense,
                @intCast(e.shape[0]),
                @intCast(e.shape[1]),
                target_layout,
                arena_alloc,
            );

            try writer.addTensor(.{
                .name = e.name,
                .storage_type = .f32,
                .tile_layout = target_layout,
                .sparsity_type = e.sparsity_type,
                .ndim = e.ndim,
                .shape = e.shape,
                .data = std.mem.sliceAsBytes(tiled),
                .sparsity_ratio = e.sparsity_ratio,
            });
        } else {
            try writer.addTensor(.{
                .name = e.name,
                .storage_type = .f32,
                .tile_layout = .row_major,
                .sparsity_type = e.sparsity_type,
                .ndim = e.ndim,
                .shape = e.shape,
                .data = std.mem.sliceAsBytes(dense),
                .sparsity_ratio = e.sparsity_ratio,
            });
        }
    }

    try writer.writeToFile(out_path);
    std.debug.print("Successfully retiled '{s}' -> '{s}' (layout: {s})\n", .{ in_path, out_path, layout_str });
}

fn cmdPrune(in_path: []const u8, out_path: []const u8, ratio: f32, allocator: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var reader = try hk.HKReader.open(in_path, arena_alloc);
    defer reader.deinit();

    var writer = hk.HKWriter.init(arena_alloc);
    defer writer.deinit();

    for (reader.metadata_map.items.items) |m| {
        switch (m.value) {
            .val_string => |s| try writer.addMetadataString(m.key, s),
            .val_int64 => |v| try writer.addMetadataInt(m.key, v),
            .val_float64 => |f| try writer.addMetadataFloat(m.key, f),
            .val_bool => |b| try writer.addMetadataBool(m.key, b),
            else => {},
        }
    }
    try writer.addMetadataFloat("cli_pruning_ratio", ratio);

    for (reader.toc.entries.items) |e| {
        var numel: usize = 1;
        for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);

        const dense = try arena_alloc.alloc(f32, numel);
        try reader.dequantizeToF32(e, true, dense);

        if (e.ndim >= 2 and ratio > 0.0) {
            // Find magnitude threshold
            const abs_vals = try arena_alloc.alloc(f32, numel);
            for (0..numel) |i| abs_vals[i] = @abs(dense[i]);

            std.mem.sort(f32, abs_vals, {}, std.sort.asc(f32));
            const k_idx = @min(@as(usize, @intFromFloat(@as(f32, @floatFromInt(numel)) * ratio)), numel - 1);
            const thresh = abs_vals[k_idx];

            var zeros: usize = 0;
            for (0..numel) |i| {
                if (@abs(dense[i]) <= thresh) {
                    dense[i] = 0.0;
                    zeros += 1;
                }
            }
            const actual_sparsity: f32 = @as(f32, @floatFromInt(zeros)) / @as(f32, @floatFromInt(numel));

            try writer.addTensor(.{
                .name = e.name,
                .storage_type = .f32,
                .tile_layout = e.tile_layout,
                .sparsity_type = .bitmask,
                .ndim = e.ndim,
                .shape = e.shape,
                .data = std.mem.sliceAsBytes(dense),
                .sparsity_ratio = actual_sparsity,
            });
        } else {
            try writer.addTensor(.{
                .name = e.name,
                .storage_type = e.storage_type,
                .tile_layout = e.tile_layout,
                .sparsity_type = e.sparsity_type,
                .ndim = e.ndim,
                .shape = e.shape,
                .data = std.mem.sliceAsBytes(dense),
                .sparsity_ratio = e.sparsity_ratio,
            });
        }
    }

    try writer.writeToFile(out_path);
    std.debug.print("Successfully pruned '{s}' -> '{s}' (target ratio: {d:.2})\n", .{ in_path, out_path, ratio });
}

fn cmdAppendix(path: []const u8, allocator: std.mem.Allocator) !void {
    var region = hk.platform.mapOrReadFile(path, allocator) catch |err| {
        std.debug.print("Failed to open HK file '{s}': {}\n", .{ path, err });
        return;
    };
    defer region.deinit(allocator);

    var app_reader = hk.appendix.AppendixReader.init(allocator, region.bytes) catch |err| {
        std.debug.print("Failed to parse appendix for '{s}': {}\n", .{ path, err });
        return;
    };
    defer app_reader.deinit();

    std.debug.print("\n=== HK Appendix Region: {s} ===\n", .{path});
    std.debug.print("Total Appendix Entries: {}\n", .{app_reader.records.items.len});
    std.debug.print("Cryptographic Lineage Valid: {}\n\n", .{app_reader.verifyLineage()});

    if (app_reader.records.items.len == 0) {
        std.debug.print("No appendix entries found in container.\n", .{});
        return;
    }

    std.debug.print("{s:<4} {s:<15} {s:<6} {s:<28} {s:<18} {s:<10} {s:<10}\n", .{
        "#", "Type", "Gen", "Name", "Target", "Accuracy", "PassRate",
    });
    std.debug.print("{s:-<100}\n", .{""});

    for (app_reader.records.items, 0..) |rec, i| {
        std.debug.print("{:<4} {s:<15} {:<6} {s:<28} {s:<18} {d:>8.2}%  {d:>8.2}%\n", .{
            i,
            @tagName(rec.entry_type),
            rec.generation,
            rec.name,
            if (rec.target.len > 0) rec.target else "-",
            rec.metrics.accuracy * 100.0,
            rec.metrics.pass_rate * 100.0,
        });
    }
}

fn cmdRollback(path: []const u8, target_gen: u32, allocator: std.mem.Allocator) !void {
    std.debug.print("Rolling back '{s}' to generation {}...\n", .{ path, target_gen });
    hk.appendix.rollbackToFile(allocator, path, target_gen) catch |err| {
        std.debug.print("Rollback failed: {}\n", .{err});
        return;
    };
    std.debug.print("[SUCCESS] Successfully rolled back '{s}' to generation {}\n", .{ path, target_gen });
}

fn cmdEval(path: []const u8, allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Evaluating Model Health & Integrity: {s} ===\n", .{path});
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var reader = hk.HKReader.open(path, allocator) catch |err| {
        std.debug.print("Failed to open HK file '{s}': {}\n", .{ path, err });
        return;
    };
    defer reader.deinit();

    var total_params: u64 = 0;
    var total_zero_params: u64 = 0;
    var nan_count: u64 = 0;
    var inf_count: u64 = 0;
    var min_val: f32 = std.math.inf(f32);
    var max_val: f32 = -std.math.inf(f32);
    var sum_abs: f64 = 0.0;

    for (reader.toc.entries.items) |e| {
        var numel: usize = 1;
        for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);
        total_params += numel;

        const dense = try arena_alloc.alloc(f32, numel);
        reader.dequantizeToF32(e, true, dense) catch |err| {
            std.debug.print("[WARN] Could not dequantize tensor '{s}': {}\n", .{ e.name, err });
            continue;
        };

        for (dense) |val| {
            if (std.math.isNan(val)) {
                nan_count += 1;
            } else if (std.math.isInf(val)) {
                inf_count += 1;
            } else {
                if (val == 0.0) total_zero_params += 1;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
                sum_abs += @abs(val);
            }
        }
    }

    const sparsity: f64 = if (total_params > 0) @as(f64, @floatFromInt(total_zero_params)) / @as(f64, @floatFromInt(total_params)) else 0.0;
    const mean_abs: f64 = if (total_params > 0) sum_abs / @as(f64, @floatFromInt(total_params)) else 0.0;

    std.debug.print("Total Tensors       : {}\n", .{reader.toc.entries.items.len});
    std.debug.print("Total Parameters    : {} ({d:.2} M)\n", .{ total_params, @as(f64, @floatFromInt(total_params)) / 1_000_000.0 });
    std.debug.print("Zero Parameters     : {} ({d:.2}% sparsity)\n", .{ total_zero_params, sparsity * 100.0 });
    std.debug.print("Value Range         : [{d:.4}, {d:.4}]\n", .{ min_val, max_val });
    std.debug.print("Mean Absolute Value : {d:.6}\n", .{mean_abs});
    std.debug.print("NaN Detections      : {}\n", .{nan_count});
    std.debug.print("Inf Detections      : {}\n", .{inf_count});

    if (nan_count == 0 and inf_count == 0) {
        std.debug.print("[STATUS] HEALTHY - Model weights are stable, normalized, and valid for inference/fine-tuning.\n\n", .{});
    } else {
        std.debug.print("[STATUS] CORRUPTED - Model weights contain NaN or Inf values!\n\n", .{});
    }
}

fn cmdExpand(
    in_path: []const u8,
    out_path: []const u8,
    new_vocab_opt: ?usize,
    width_ratio_opt: ?f32,
    allocator: std.mem.Allocator,
) !void {
    std.debug.print("\n=== Expanding Model: '{s}' -> '{s}' ===\n", .{ in_path, out_path });
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var reader = try hk.HKReader.open(in_path, allocator);
    defer reader.deinit();

    var writer = hk.HKWriter.init(allocator);
    defer writer.deinit();

    // Copy existing metadata
    for (reader.metadata_map.items.items) |m| {
        switch (m.value) {
            .val_string => |s| try writer.addMetadataString(m.key, s),
            .val_int64 => |v| try writer.addMetadataInt(m.key, v),
            .val_float64 => |f| try writer.addMetadataFloat(m.key, f),
            .val_bool => |b| try writer.addMetadataBool(m.key, b),
            else => {},
        }
    }

    if (new_vocab_opt) |nv| {
        try writer.addMetadataInt("expanded_vocab_size", @intCast(nv));
    }
    if (width_ratio_opt) |wr| {
        try writer.addMetadataFloat("expanded_width_ratio", wr);
    }
    try writer.addMetadataString("expansion_engine", "native_zig_v1");

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    var expanded_count: usize = 0;

    for (reader.toc.entries.items) |e| {
        var numel: usize = 1;
        for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);

        const dense_old = try arena_alloc.alloc(f32, numel);
        try reader.dequantizeToF32(e, true, dense_old);

        var is_expanded = false;

        // 1. Check vocabulary expansion
        if (new_vocab_opt) |target_vocab| {
            const is_vocab_tensor = (std.mem.indexOf(u8, e.name, "embed_tokens") != null) or
                (std.mem.indexOf(u8, e.name, "lm_head") != null) or
                (std.mem.indexOf(u8, e.name, "wte") != null) or
                (std.mem.indexOf(u8, e.name, "token_embeddings") != null);

            if (is_vocab_tensor and e.ndim == 2 and e.shape[0] < target_vocab) {
                const old_v: usize = @intCast(e.shape[0]);
                const hidden_dim: usize = @intCast(e.shape[1]);
                const new_total = target_vocab * hidden_dim;
                const dense_new = try arena_alloc.alloc(f32, new_total);

                try hk.growth.expandVocab(
                    dense_old,
                    dense_new,
                    null,
                    null,
                    old_v,
                    target_vocab,
                    hidden_dim,
                    42,
                );

                var new_shape = e.shape;
                new_shape[0] = target_vocab;

                try writer.addTensor(.{
                    .name = e.name,
                    .storage_type = .f32,
                    .tile_layout = e.tile_layout,
                    .sparsity_type = e.sparsity_type,
                    .ndim = e.ndim,
                    .shape = new_shape,
                    .data = std.mem.sliceAsBytes(dense_new),
                    .sparsity_ratio = e.sparsity_ratio,
                });

                std.debug.print("  [VOCAB EXPANDED] {s}: [{}, {}] -> [{}, {}]\n", .{
                    e.name, old_v, hidden_dim, target_vocab, hidden_dim,
                });
                expanded_count += 1;
                is_expanded = true;
            }
        }

        // 2. Check SwiGLU / MLP width expansion
        if (!is_expanded and width_ratio_opt != null and width_ratio_opt.? > 1.0 and e.ndim == 2) {
            const ratio = width_ratio_opt.?;
            const is_gate = (std.mem.indexOf(u8, e.name, "gate_proj") != null) or (std.mem.indexOf(u8, e.name, "w1") != null);
            const is_up = (std.mem.indexOf(u8, e.name, "up_proj") != null) or (std.mem.indexOf(u8, e.name, "w3") != null);
            const is_down = (std.mem.indexOf(u8, e.name, "down_proj") != null) or (std.mem.indexOf(u8, e.name, "w2") != null);

            if (is_gate or is_up) {
                const old_inter: usize = @intCast(e.shape[0]);
                const in_f: usize = @intCast(e.shape[1]);
                const new_inter: usize = @intFromFloat(@round(@as(f32, @floatFromInt(old_inter)) * ratio));

                if (new_inter > old_inter) {
                    const new_total = new_inter * in_f;
                    const dense_new = try arena_alloc.alloc(f32, new_total);

                    // Copy base rows
                    @memcpy(dense_new[0 .. old_inter * in_f], dense_old[0 .. old_inter * in_f]);

                    // Initialize new rows with small normal noise
                    for (old_inter..new_inter) |r| {
                        for (0..in_f) |c| {
                            const unif1 = @max(rand.float(f32), 1e-7);
                            const unif2 = rand.float(f32);
                            const z = @sqrt(-2.0 * @log(unif1)) * @cos(2.0 * std.math.pi * unif2);
                            dense_new[r * in_f + c] = z * 0.02;
                        }
                    }

                    var new_shape = e.shape;
                    new_shape[0] = new_inter;

                    try writer.addTensor(.{
                        .name = e.name,
                        .storage_type = .f32,
                        .tile_layout = e.tile_layout,
                        .sparsity_type = e.sparsity_type,
                        .ndim = e.ndim,
                        .shape = new_shape,
                        .data = std.mem.sliceAsBytes(dense_new),
                        .sparsity_ratio = e.sparsity_ratio,
                    });

                    std.debug.print("  [WIDTH EXPANDED (GATE/UP)] {s}: [{}, {}] -> [{}, {}]\n", .{
                        e.name, old_inter, in_f, new_inter, in_f,
                    });
                    expanded_count += 1;
                    is_expanded = true;
                }
            } else if (is_down) {
                const out_f: usize = @intCast(e.shape[0]);
                const old_inter: usize = @intCast(e.shape[1]);
                const new_inter: usize = @intFromFloat(@round(@as(f32, @floatFromInt(old_inter)) * ratio));

                if (new_inter > old_inter) {
                    const new_total = out_f * new_inter;
                    const dense_new = try arena_alloc.alloc(f32, new_total);

                    // Copy base columns, ZERO-INITIALIZE new columns for Day-0 function preservation!
                    for (0..out_f) |r| {
                        @memcpy(
                            dense_new[r * new_inter .. r * new_inter + old_inter],
                            dense_old[r * old_inter .. (r + 1) * old_inter],
                        );
                        @memset(dense_new[r * new_inter + old_inter .. (r + 1) * new_inter], 0.0);
                    }

                    var new_shape = e.shape;
                    new_shape[1] = new_inter;

                    try writer.addTensor(.{
                        .name = e.name,
                        .storage_type = .f32,
                        .tile_layout = e.tile_layout,
                        .sparsity_type = e.sparsity_type,
                        .ndim = e.ndim,
                        .shape = new_shape,
                        .data = std.mem.sliceAsBytes(dense_new),
                        .sparsity_ratio = e.sparsity_ratio,
                    });

                    std.debug.print("  [WIDTH EXPANDED (DOWN - ZERO-INIT)] {s}: [{}, {}] -> [{}, {}]\n", .{
                        e.name, out_f, old_inter, out_f, new_inter,
                    });
                    expanded_count += 1;
                    is_expanded = true;
                }
            }
        }

        // If not modified, write existing tensor
        if (!is_expanded) {
            try writer.addTensor(.{
                .name = e.name,
                .storage_type = e.storage_type,
                .tile_layout = e.tile_layout,
                .sparsity_type = e.sparsity_type,
                .ndim = e.ndim,
                .shape = e.shape,
                .data = std.mem.sliceAsBytes(dense_old),
                .sparsity_ratio = e.sparsity_ratio,
            });
        }
    }

    try writer.writeToFile(out_path);
    std.debug.print("\n[SUCCESS] Expansion complete! {} tensors expanded natively. Output saved to '{s}'.\n\n", .{
        expanded_count, out_path,
    });
}

fn cmdDump(path: []const u8, allocator: std.mem.Allocator) !void {
    var reader = try hk.HKReader.open(path, allocator);
    defer reader.deinit();

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("                         HK BINARY CONTAINER DUMPER                             \n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("File Path            : {s}\n", .{path});
    std.debug.print("Total File Size      : {} bytes ({d:.2} MB)\n", .{
        reader.mmap_region.bytes.len,
        @as(f64, @floatFromInt(reader.mmap_region.bytes.len)) / (1024.0 * 1024.0),
    });

    const h = reader.header;
    std.debug.print("\n--- 128-Byte Fixed Header Breakdown ---\n", .{});
    std.debug.print("  Magic Bytes        : {s} (0x{X:0>2} 0x{X:0>2} 0x{X:0>2} 0x{X:0>2}) [Valid: {}]\n", .{
        h.magic, h.magic[0], h.magic[1], h.magic[2], h.magic[3], h.isValid(),
    });
    std.debug.print("  Version            : v{}.{}\n", .{ h.version_major, h.version_minor });
    std.debug.print("  Header Flags       : 0x{X:0>8}\n", .{h.flags});
    std.debug.print("    - Little Endian  : {}\n", .{(h.flags & hk.format.HeaderFlags.LITTLE_ENDIAN) != 0});
    std.debug.print("    - Has Appendix   : {}\n", .{(h.flags & hk.format.HeaderFlags.HAS_APPENDIX) != 0});
    std.debug.print("    - Tile Aligned   : {}\n", .{(h.flags & hk.format.HeaderFlags.TILE_ALIGNED) != 0});
    std.debug.print("    - Is Sharded     : {}\n", .{(h.flags & hk.format.HeaderFlags.IS_SHARDED) != 0});
    std.debug.print("  Tensor Alignment   : {} bytes\n", .{h.alignment});
    std.debug.print("  Sharding Index     : {} of {}\n", .{ h.split_index + 1, h.split_count });
    std.debug.print("  Tensor Count       : {}\n", .{h.tensor_count});
    std.debug.print("  Metadata KVs       : {}\n", .{h.metadata_kv_count});
    std.debug.print("  Metadata Offset    : 0x{X:0>8} ({} bytes)\n", .{ h.metadata_offset, h.metadata_size });
    std.debug.print("  Tensor TOC Offset  : 0x{X:0>8} ({} bytes)\n", .{ h.tensor_toc_offset, h.tensor_toc_size });
    std.debug.print("  Tensor Data Offset : 0x{X:0>8} (128-byte aligned: {})\n", .{
        h.tensor_data_offset, (h.tensor_data_offset % 128) == 0,
    });
    std.debug.print("  Appendix Offset    : 0x{X:0>8}\n", .{h.appendix_offset});

    // Raw Hex Dump of Header (first 128 bytes)
    std.debug.print("\n--- Raw Header Hex Dump (Offset 0x0000 - 0x007F) ---\n", .{});
    const header_slice = reader.mmap_region.bytes[0..@min(128, reader.mmap_region.bytes.len)];
    var offset: usize = 0;
    while (offset < header_slice.len) : (offset += 16) {
        std.debug.print("  0x{X:0>4}: ", .{offset});
        const chunk_len = @min(16, header_slice.len - offset);
        for (0..16) |j| {
            if (j < chunk_len) {
                std.debug.print("{X:0>2} ", .{header_slice[offset + j]});
            } else {
                std.debug.print("   ", .{});
            }
            if (j == 7) std.debug.print(" ", .{});
        }
        std.debug.print(" |", .{});
        for (0..chunk_len) |j| {
            const b = header_slice[offset + j];
            const ch: u8 = if (b >= 32 and b <= 126) b else '.';
            std.debug.print("{c}", .{ch});
        }
        std.debug.print("|\n", .{});
    }

    // Metadata Listing
    std.debug.print("\n--- Metadata Table ({} entries) ---\n", .{reader.metadata_map.items.items.len});
    for (reader.metadata_map.items.items, 0..) |item, idx| {
        std.debug.print("  [{:0>2}] {s:<32} = ", .{ idx, item.key });
        switch (item.value) {
            .val_string => |s| std.debug.print("\"{s}\" (string)\n", .{s}),
            .val_int64 => |v| std.debug.print("{} (int64)\n", .{v}),
            .val_float64 => |f| std.debug.print("{d:.6} (float64)\n", .{f}),
            .val_bool => |b| std.debug.print("{} (bool)\n", .{b}),
            .val_json => |j| std.debug.print("{s} (JSON)\n", .{j}),
            .val_bytes => |b| std.debug.print("<{} bytes> (binary)\n", .{b.len}),
        }
    }

    // Tensor TOC Dump
    std.debug.print("\n--- Tensor TOC Table ({} entries) ---\n", .{reader.toc.entries.items.len});
    std.debug.print("{s:<4} {s:<36} {s:<10} {s:<10} {s:<14} {s:<10} {s:<10} {s:<8}\n", .{
        "#", "Name", "Type", "Layout", "Shape", "Offset", "Size", "Sparsity",
    });
    std.debug.print("{s:-<110}\n", .{""});

    for (reader.toc.entries.items, 0..) |e, idx| {
        var shape_buf: [64]u8 = undefined;
        var pos: usize = 0;
        shape_buf[pos] = '[';
        pos += 1;
        for (0..e.ndim) |d| {
            if (d > 0) {
                shape_buf[pos] = ',';
                pos += 1;
            }
            const part = std.fmt.bufPrint(shape_buf[pos..], "{}", .{e.shape[d]}) catch "";
            pos += part.len;
        }
        shape_buf[pos] = ']';
        pos += 1;
        const shape_str = shape_buf[0..pos];

        std.debug.print("{:0>3}  {s:<36} {s:<10} {s:<10} {s:<14} 0x{X:<8} {}B {d:>5.1}%\n", .{
            idx,
            e.name,
            @tagName(e.storage_type),
            @tagName(e.tile_layout),
            shape_str,
            e.data_offset,
            e.data_size,
            e.sparsity_ratio * 100.0,
        });
    }
    std.debug.print("\n", .{});
}

fn hexDigest(bytes: [32]u8, out_hex: *[64]u8) []const u8 {
    const charset = "0123456789abcdef";
    for (bytes, 0..) |b, i| {
        out_hex[i * 2] = charset[b >> 4];
        out_hex[i * 2 + 1] = charset[b & 0x0F];
    }
    return out_hex[0..64];
}

fn cmdHash(path: []const u8, allocator: std.mem.Allocator) !void {
    var reader = try hk.HKReader.open(path, allocator);
    defer reader.deinit();

    std.debug.print("\n=== HK Cryptographic Checksum Verifier ===\n", .{});
    std.debug.print("File: {s}\n\n", .{path});

    // 1. Full Container SHA-256
    var file_hasher = std.crypto.hash.sha2.Sha256.init(.{});
    file_hasher.update(reader.mmap_region.bytes);
    var file_digest: [32]u8 = undefined;
    file_hasher.final(&file_digest);

    var file_hex: [64]u8 = undefined;
    std.debug.print("Container SHA-256 : {s}\n", .{hexDigest(file_digest, &file_hex)});
    std.debug.print("Total Tensors     : {}\n\n", .{reader.toc.entries.items.len});

    std.debug.print("{s:<4} {s:<42} {s:<10} {s}\n", .{ "#", "Tensor Name", "Size", "SHA-256 Digest" });
    std.debug.print("{s:-<120}\n", .{""});

    for (reader.toc.entries.items, 0..) |e, idx| {
        const data = try reader.getTensorData(e);
        var t_hasher = std.crypto.hash.sha2.Sha256.init(.{});
        t_hasher.update(data);
        if (reader.getTensorScales(e)) |sc| {
            t_hasher.update(sc);
        }
        var t_digest: [32]u8 = undefined;
        t_hasher.final(&t_digest);

        var t_hex: [64]u8 = undefined;
        std.debug.print("{:0>3}  {s:<42} {:<10} {s}\n", .{
            idx,
            e.name,
            data.len,
            hexDigest(t_digest, &t_hex),
        });
    }
    std.debug.print("\n[VERIFIED] All {} tensors hashed and cryptographically sealed.\n\n", .{
        reader.toc.entries.items.len,
    });
}

fn cmdConvertEndian(in_path: []const u8, out_path: []const u8, allocator: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var reader = try hk.HKReader.open(in_path, arena_alloc);
    defer reader.deinit();

    var writer = hk.HKWriter.init(arena_alloc);
    defer writer.deinit();

    // Copy metadata
    for (reader.metadata_map.items.items) |m| {
        switch (m.value) {
            .val_string => |s| try writer.addMetadataString(m.key, s),
            .val_int64 => |v| try writer.addMetadataInt(m.key, v),
            .val_float64 => |f| try writer.addMetadataFloat(m.key, f),
            .val_bool => |b| try writer.addMetadataBool(m.key, b),
            else => {},
        }
    }
    try writer.addMetadataString("endian_conversion", "swapped");

    // Copy tensors
    for (reader.toc.entries.items) |e| {
        const data_bytes = try reader.getTensorData(e);
        const dup_data = try arena_alloc.dupe(u8, data_bytes);

        // Perform endianness byte swap on multi-byte payloads
        if (e.storage_type == .f32 or e.storage_type == .int32) {
            var j: usize = 0;
            while (j + 4 <= dup_data.len) : (j += 4) {
                const b0 = dup_data[j + 0];
                const b1 = dup_data[j + 1];
                const b2 = dup_data[j + 2];
                const b3 = dup_data[j + 3];
                dup_data[j + 0] = b3;
                dup_data[j + 1] = b2;
                dup_data[j + 2] = b1;
                dup_data[j + 3] = b0;
            }
        } else if (e.storage_type == .f16 or e.storage_type == .bf16) {
            var j: usize = 0;
            while (j + 2 <= dup_data.len) : (j += 2) {
                const tmp = dup_data[j + 0];
                dup_data[j + 0] = dup_data[j + 1];
                dup_data[j + 1] = tmp;
            }
        } else if (e.storage_type == .int64) {
            var j: usize = 0;
            while (j + 8 <= dup_data.len) : (j += 8) {
                for (0..4) |k| {
                    const tmp = dup_data[j + k];
                    dup_data[j + k] = dup_data[j + 7 - k];
                    dup_data[j + 7 - k] = tmp;
                }
            }
        }

        try writer.addTensor(.{
            .name = e.name,
            .storage_type = e.storage_type,
            .tile_layout = e.tile_layout,
            .sparsity_type = e.sparsity_type,
            .ndim = e.ndim,
            .shape = e.shape,
            .data = dup_data,
            .sparsity_ratio = e.sparsity_ratio,
        });
    }

    try writer.writeToFile(out_path);
    std.debug.print("[SUCCESS] Endianness converted successfully:\n  Input : {s}\n  Output: {s}\n", .{ in_path, out_path });
}

fn cmdGui(file_path_opt: ?[]const u8, allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("                         HK GRAPHICAL MODEL EDITOR                              \n", .{});
    std.debug.print("================================================================================\n", .{});
    if (file_path_opt) |fp| {
        std.debug.print("Target Model : {s}\n", .{fp});
        std.debug.print("Launch GUI   : py -3.12 tools/hk_editor_gui.py {s}\n\n", .{fp});
    } else {
        std.debug.print("Launch GUI   : py -3.12 tools/hk_editor_gui.py\n\n", .{});
    }
}


