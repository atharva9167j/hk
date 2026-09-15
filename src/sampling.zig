const std = @import("std");

pub const SamplingParams = struct {
    temperature: f32 = 0.7,
    top_k: usize = 40,
    top_p: f32 = 0.9,
    min_p: f32 = 0.05,
    repetition_penalty: f32 = 1.1,
    seed: u64 = 42,
};

pub const TokenProb = struct {
    id: u32,
    prob: f32,
};

pub const Sampler = struct {
    prng: std.Random.DefaultPrng,

    pub fn init(seed: u64) Sampler {
        return .{
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    /// Pure greedy argmax sampling
    pub fn sampleGreedy(logits: []const f32) u32 {
        if (logits.len == 0) return 0;
        var best_idx: u32 = 0;
        var best_val: f32 = logits[0];
        for (logits, 0..) |val, i| {
            if (val > best_val) {
                best_val = val;
                best_idx = @intCast(i);
            }
        }
        return best_idx;
    }

    /// Full sampling pipeline with repetition penalty, temperature, top-k, top-p, and min-p
    pub fn sample(
        self: *Sampler,
        allocator: std.mem.Allocator,
        logits: []f32,
        params: SamplingParams,
        history: []const u32,
    ) !u32 {
        if (params.temperature <= 0.0) {
            return sampleGreedy(logits);
        }

        const vocab_size = logits.len;
        if (vocab_size == 0) return 0;

        // 1. Apply repetition penalty
        if (params.repetition_penalty != 1.0 and history.len > 0) {
            for (history) |tid| {
                if (tid < vocab_size) {
                    if (logits[tid] > 0.0) {
                        logits[tid] /= params.repetition_penalty;
                    } else {
                        logits[tid] *= params.repetition_penalty;
                    }
                }
            }
        }

        // 2. Scale by temperature
        const inv_temp = 1.0 / params.temperature;
        for (logits) |*l| {
            l.* *= inv_temp;
        }

        // 3. Find max logit for numerical stability
        var max_logit: f32 = logits[0];
        for (logits[1..]) |l| {
            if (l > max_logit) max_logit = l;
        }

        // Fast path for top-k <= 256 (avoids allocating and sorting entire vocab)
        if (params.top_k > 0 and params.top_k <= 256) {
            const k = @min(params.top_k, vocab_size);
            var top_k_buf: [256]TokenProb = undefined;
            var count: usize = 0;

            for (logits, 0..) |l, i| {
                const exp_val = @exp(l - max_logit);
                if (count < k) {
                    var ins_idx = count;
                    while (ins_idx > 0 and top_k_buf[ins_idx - 1].prob < exp_val) : (ins_idx -= 1) {
                        top_k_buf[ins_idx] = top_k_buf[ins_idx - 1];
                    }
                    top_k_buf[ins_idx] = .{ .id = @intCast(i), .prob = exp_val };
                    count += 1;
                } else if (exp_val > top_k_buf[k - 1].prob) {
                    var ins_idx = k - 1;
                    while (ins_idx > 0 and top_k_buf[ins_idx - 1].prob < exp_val) : (ins_idx -= 1) {
                        top_k_buf[ins_idx] = top_k_buf[ins_idx - 1];
                    }
                    top_k_buf[ins_idx] = .{ .id = @intCast(i), .prob = exp_val };
                }
            }

            var sum: f32 = 0.0;
            for (top_k_buf[0..count]) |c| {
                sum += c.prob;
            }
            if (sum <= 0.0 or std.math.isNan(sum)) {
                return sampleGreedy(logits);
            }
            const inv_sum = 1.0 / sum;
            for (top_k_buf[0..count]) |*c| {
                c.prob *= inv_sum;
            }

            // Min-P filtering
            var valid_count: usize = count;
            if (params.min_p > 0.0 and count > 0) {
                const min_p_threshold = top_k_buf[0].prob * params.min_p;
                var vc: usize = 0;
                for (top_k_buf[0..count]) |c| {
                    if (c.prob >= min_p_threshold) {
                        top_k_buf[vc] = c;
                        vc += 1;
                    }
                }
                valid_count = if (vc > 0) vc else 1;
            }

            // Top-P (nucleus) truncation
            if (params.top_p < 1.0) {
                var cum_sum: f32 = 0.0;
                var cut_idx: usize = 1;
                while (cut_idx < valid_count) : (cut_idx += 1) {
                    cum_sum += top_k_buf[cut_idx - 1].prob;
                    if (cum_sum >= params.top_p) break;
                }
                valid_count = cut_idx;
            }

            var filtered_sum: f32 = 0.0;
            for (top_k_buf[0..valid_count]) |c| {
                filtered_sum += c.prob;
            }
            if (filtered_sum <= 0.0) return top_k_buf[0].id;

            const rand_val = self.prng.random().float(f32) * filtered_sum;
            var running: f32 = 0.0;
            for (top_k_buf[0..valid_count]) |c| {
                running += c.prob;
                if (running >= rand_val) return c.id;
            }
            return top_k_buf[0].id;
        }

        // General path for unbounded top-k / large vocabs
        var candidates = try allocator.alloc(TokenProb, vocab_size);
        defer allocator.free(candidates);

        var sum: f32 = 0.0;
        for (logits, 0..) |l, i| {
            const exp_val = @exp(l - max_logit);
            candidates[i] = .{
                .id = @intCast(i),
                .prob = exp_val,
            };
            sum += exp_val;
        }

        if (sum <= 0.0 or std.math.isNan(sum)) {
            return sampleGreedy(logits);
        }

        const inv_sum = 1.0 / sum;
        for (candidates) |*c| {
            c.prob *= inv_sum;
        }

        // 5. Min-P filtering
        var valid_count: usize = 0;
        if (params.min_p > 0.0) {
            var max_p: f32 = 0.0;
            for (candidates) |c| {
                if (c.prob > max_p) max_p = c.prob;
            }
            const min_p_threshold = max_p * params.min_p;
            for (candidates) |c| {
                if (c.prob >= min_p_threshold) {
                    candidates[valid_count] = c;
                    valid_count += 1;
                }
            }
        } else {
            valid_count = candidates.len;
        }

        if (valid_count == 0) return sampleGreedy(logits);

        // 6. Sort candidates descending by probability
        std.mem.sort(TokenProb, candidates[0..valid_count], {}, struct {
            fn greaterThan(_: void, a: TokenProb, b: TokenProb) bool {
                return a.prob > b.prob;
            }
        }.greaterThan);

        // 7. Top-K truncation
        if (params.top_k > 0 and params.top_k < valid_count) {
            valid_count = params.top_k;
        }

        // 8. Top-P (nucleus) truncation
        if (params.top_p < 1.0) {
            var cum_sum: f32 = 0.0;
            var cut_idx: usize = 1;
            while (cut_idx < valid_count) : (cut_idx += 1) {
                cum_sum += candidates[cut_idx - 1].prob;
                if (cum_sum >= params.top_p) {
                    break;
                }
            }
            valid_count = cut_idx;
        }

        // 9. Renormalize remaining probabilities
        var filtered_sum: f32 = 0.0;
        for (candidates[0..valid_count]) |c| {
            filtered_sum += c.prob;
        }

        if (filtered_sum <= 0.0) {
            return candidates[0].id;
        }

        // 10. Sample randomly from categorical distribution
        const rand_val = self.prng.random().float(f32) * filtered_sum;
        var running: f32 = 0.0;
        for (candidates[0..valid_count]) |c| {
            running += c.prob;
            if (running >= rand_val) {
                return c.id;
            }
        }

        return candidates[0].id;
    }
};
