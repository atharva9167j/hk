//! Standalone pp/tg throughput benchmark for the CPU TransformerEngine,
//! matching llama.cpp's llama-bench methodology: prompt processing (pp) is
//! timed as a batch of forward() calls over synthetic prompt tokens, token
//! generation (tg) as a batch of forward() calls feeding a fixed token back
//! in (no sampling/decoding -- this measures compute throughput only, not
//! output quality).

const std = @import("std");
const hk = @import("hk");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer it.deinit();
    _ = it.next();
    const path = it.next() orelse {
        std.debug.print("usage: hk_cpu_bench <file.hk> [pp_tokens] [tg_tokens]\n", .{});
        return error.MissingArgument;
    };
    const pp_n: usize = if (it.next()) |s| try std.fmt.parseInt(usize, s, 10) else 512;
    const tg_n: usize = if (it.next()) |s| try std.fmt.parseInt(usize, s, 10) else 128;

    std.debug.print("Loading model: {s}\n", .{path});
    var reader = try hk.HKReader.open(path, allocator);
    defer reader.deinit();

    var engine = try hk.inference.TransformerEngine.initFromReader(allocator, &reader);
    defer engine.deinit();
    defer allocator.destroy(engine);

    std.debug.print("layers={d} dim={d} n_heads={d} n_kv_heads={d} vocab={d}\n", .{
        engine.config.n_layers, engine.config.dim, engine.config.n_heads,
        engine.config.n_kv_heads, engine.config.vocab_size,
    });

    const io = std.Options.debug_io;

    // Prompt processing: pp_n synthetic tokens through forward(), sequential
    // positions -- same shape of work as prefilling a real prompt (token ids
    // don't matter for throughput, only position/KV-cache growth do).
    const pp_start = std.Io.Clock.Timestamp.now(io, .awake);
    var pos: usize = 0;
    while (pos < pp_n) : (pos += 1) {
        const tok: u32 = @intCast(pos % @max(engine.config.vocab_size, 1));
        _ = engine.forward(tok, pos);
    }
    const pp_ns: i96 = pp_start.untilNow(io).raw.nanoseconds;
    const pp_s = @as(f64, @floatFromInt(pp_ns)) / 1e9;

    // Token generation: tg_n more forward() calls, one per position, exactly
    // like decode steps (KV-cache keeps growing across the pp_n prefill).
    const tg_start = std.Io.Clock.Timestamp.now(io, .awake);
    var gen: usize = 0;
    while (gen < tg_n and pos < engine.config.max_seq_len) : ({
        gen += 1;
        pos += 1;
    }) {
        const tok: u32 = @intCast(pos % @max(engine.config.vocab_size, 1));
        _ = engine.forward(tok, pos);
    }
    const tg_ns: i96 = tg_start.untilNow(io).raw.nanoseconds;
    const tg_s = @as(f64, @floatFromInt(tg_ns)) / 1e9;

    std.debug.print(
        "\n== CPU pp/tg benchmark ==\n" ++
            "  pp: {d} tokens in {d:.3} s -> {d:.2} tok/s ({d:.2} ms/tok)\n" ++
            "  tg: {d} tokens in {d:.3} s -> {d:.2} tok/s ({d:.2} ms/tok)\n",
        .{
            pp_n,       pp_s, @as(f64, @floatFromInt(pp_n)) / pp_s,       (pp_s * 1000.0) / @as(f64, @floatFromInt(pp_n)),
            gen,        tg_s, @as(f64, @floatFromInt(gen)) / tg_s,        (tg_s * 1000.0) / @as(f64, @floatFromInt(gen)),
        },
    );
}
