const std = @import("std");

pub const ArchFamily = enum {
    llama,
    llama4,
    qwen2,
    qwen3,
    qwen2_moe,
    qwen3_moe,
    mistral,
    mixtral,
    deepseek,
    deepseek2,
    deepseek3,
    deepseek_r1,
    gemma,
    gemma2,
    grok,
    falcon,
    phi,
    phi2,
    phi3,
    phi4,
    dbrx,
    command_r,
    mamba,
    mamba2,
    whisper,
    flux,
    modern_bert,
    generic,

    pub fn asString(self: ArchFamily) []const u8 {
        return switch (self) {
            .llama => "llama",
            .llama4 => "llama4",
            .qwen2 => "qwen2",
            .qwen3 => "qwen3",
            .qwen2_moe => "qwen2_moe",
            .qwen3_moe => "qwen3_moe",
            .mistral => "mistral",
            .mixtral => "mixtral",
            .deepseek => "deepseek",
            .deepseek2 => "deepseek2",
            .deepseek3 => "deepseek3",
            .deepseek_r1 => "deepseek_r1",
            .gemma => "gemma",
            .gemma2 => "gemma2",
            .grok => "grok",
            .falcon => "falcon",
            .phi => "phi",
            .phi2 => "phi2",
            .phi3 => "phi3",
            .phi4 => "phi4",
            .dbrx => "dbrx",
            .command_r => "command-r",
            .mamba => "mamba",
            .mamba2 => "mamba2",
            .whisper => "whisper",
            .flux => "flux",
            .modern_bert => "modern_bert",
            .generic => "llama",
        };
    }
};

pub fn canonicalizeArch(name: []const u8) []const u8 {
    var lower_buf: [128]u8 = undefined;
    const len = @min(name.len, lower_buf.len);
    for (0..len) |i| {
        lower_buf[i] = std.ascii.toLower(name[i]);
    }
    const s = lower_buf[0..len];

    if (std.mem.indexOf(u8, s, "deepseek_r1") != null or std.mem.indexOf(u8, s, "deepseek-r1") != null) return "deepseek_r1";
    if (std.mem.indexOf(u8, s, "deepseekv3") != null or std.mem.indexOf(u8, s, "deepseek_v3") != null or std.mem.indexOf(u8, s, "deepseek3") != null) return "deepseek3";
    if (std.mem.indexOf(u8, s, "deepseekv2") != null or std.mem.indexOf(u8, s, "deepseek_v2") != null or std.mem.indexOf(u8, s, "deepseek2") != null) return "deepseek2";
    if (std.mem.indexOf(u8, s, "deepseek") != null) return "deepseek";
    if (std.mem.indexOf(u8, s, "llama4") != null or std.mem.indexOf(u8, s, "llama-4") != null) return "llama4";
    if (std.mem.indexOf(u8, s, "llama") != null) return "llama";
    if (std.mem.indexOf(u8, s, "qwen3_moe") != null or std.mem.indexOf(u8, s, "qwen3moe") != null) return "qwen3_moe";
    if (std.mem.indexOf(u8, s, "qwen2_moe") != null or std.mem.indexOf(u8, s, "qwen2moe") != null) return "qwen2_moe";
    if (std.mem.indexOf(u8, s, "qwen3") != null) return "qwen3";
    if (std.mem.indexOf(u8, s, "qwen2.5") != null or std.mem.indexOf(u8, s, "qwen2_5") != null or std.mem.indexOf(u8, s, "qwen2") != null or std.mem.indexOf(u8, s, "qwen") != null) return "qwen2";
    if (std.mem.indexOf(u8, s, "mixtral") != null) return "mixtral";
    if (std.mem.indexOf(u8, s, "mistral") != null) return "mistral";
    if (std.mem.indexOf(u8, s, "gemma2") != null or std.mem.indexOf(u8, s, "gemma-2") != null) return "gemma2";
    if (std.mem.indexOf(u8, s, "gemma") != null) return "gemma";
    if (std.mem.indexOf(u8, s, "grok") != null) return "grok";
    if (std.mem.indexOf(u8, s, "falcon") != null) return "falcon";
    if (std.mem.indexOf(u8, s, "phi4") != null) return "phi4";
    if (std.mem.indexOf(u8, s, "phi3") != null) return "phi3";
    if (std.mem.indexOf(u8, s, "phi2") != null) return "phi2";
    if (std.mem.indexOf(u8, s, "phi") != null) return "phi";
    if (std.mem.indexOf(u8, s, "dbrx") != null) return "dbrx";
    if (std.mem.indexOf(u8, s, "command-r") != null or std.mem.indexOf(u8, s, "cohere") != null) return "command-r";
    if (std.mem.indexOf(u8, s, "mamba2") != null) return "mamba2";
    if (std.mem.indexOf(u8, s, "mamba") != null) return "mamba";
    if (std.mem.indexOf(u8, s, "whisper") != null) return "whisper";
    if (std.mem.indexOf(u8, s, "flux") != null) return "flux";
    if (std.mem.indexOf(u8, s, "modern_bert") != null or std.mem.indexOf(u8, s, "modernbert") != null) return "modern_bert";
    if (std.mem.indexOf(u8, s, "bert") != null) return "modern_bert";

    return "llama";
}

/// Detects canonical architecture from a raw JSON config string
pub fn detectArchitectureFromJson(json_str: []const u8) []const u8 {
    // Check "model_type": "..."
    if (std.mem.indexOf(u8, json_str, "\"model_type\"")) |idx| {
        const after = json_str[idx + 12 ..];
        if (std.mem.indexOf(u8, after, "\"")) |q1| {
            const rest = after[q1 + 1 ..];
            if (std.mem.indexOf(u8, rest, "\"")) |q2| {
                const mt = rest[0..q2];
                return canonicalizeArch(mt);
            }
        }
    }
    // Check "architectures": ["..."]
    if (std.mem.indexOf(u8, json_str, "\"architectures\"")) |idx| {
        const after = json_str[idx + 15 ..];
        if (std.mem.indexOf(u8, after, "\"")) |q1| {
            const rest = after[q1 + 1 ..];
            if (std.mem.indexOf(u8, rest, "\"")) |q2| {
                const arch = rest[0..q2];
                return canonicalizeArch(arch);
            }
        }
    }
    return "llama";
}

/// Helper to parse layer number from `model.layers.<N>.` or `backbone.layers.<N>.`
fn extractLayerNum(slice: []const u8, out_num: *[]const u8, rest: *[]const u8) bool {
    var end: usize = 0;
    while (end < slice.len and slice[end] >= '0' and slice[end] <= '9') : (end += 1) {}
    if (end == 0 or end >= slice.len) return false;
    out_num.* = slice[0..end];
    rest.* = slice[end..];
    return true;
}

/// High-performance Hugging Face tensor name mapping into HK Canonical format
pub fn mapTensorNameToHk(name: []const u8, arch: []const u8, out_buf: []u8) ![]const u8 {
    // 1. Embed tokens
    if (std.mem.eql(u8, name, "model.embed_tokens.weight") or std.mem.eql(u8, name, "backbone.embeddings.weight")) {
        const target = "embed_tokens.weight";
        if (out_buf.len < target.len) return error.BufferTooSmall;
        @memcpy(out_buf[0..target.len], target);
        return out_buf[0..target.len];
    }
    // 2. Norm
    if (std.mem.eql(u8, name, "model.norm.weight") or std.mem.eql(u8, name, "backbone.norm_f.weight")) {
        const target = "norm.weight";
        if (out_buf.len < target.len) return error.BufferTooSmall;
        @memcpy(out_buf[0..target.len], target);
        return out_buf[0..target.len];
    }
    // 3. LM Head
    if (std.mem.eql(u8, name, "lm_head.weight")) {
        const target = "lm_head.weight";
        if (out_buf.len < target.len) return error.BufferTooSmall;
        @memcpy(out_buf[0..target.len], target);
        return out_buf[0..target.len];
    }

    // 4. Layers: "model.layers." or "backbone.layers."
    var prefix_len: usize = 0;
    if (std.mem.startsWith(u8, name, "model.layers.")) {
        prefix_len = 13;
    } else if (std.mem.startsWith(u8, name, "backbone.layers.")) {
        prefix_len = 16;
    }

    if (prefix_len > 0) {
        var num: []const u8 = undefined;
        var suffix: []const u8 = undefined;
        if (extractLayerNum(name[prefix_len..], &num, &suffix)) {
            // Check suffix mappings
            const mapping_entry = struct { hf: []const u8, hk: []const u8 };
            const layer_mappings = [_]mapping_entry{
                // Self Attention Projections
                .{ .hf = ".self_attn.q_proj.weight", .hk = ".attn_q.weight" },
                .{ .hf = ".self_attn.k_proj.weight", .hk = ".attn_k.weight" },
                .{ .hf = ".self_attn.v_proj.weight", .hk = ".attn_v.weight" },
                .{ .hf = ".self_attn.o_proj.weight", .hk = ".attn_output.weight" },
                .{ .hf = ".self_attn.q_proj.bias", .hk = ".attn_q.bias" },
                .{ .hf = ".self_attn.k_proj.bias", .hk = ".attn_k.bias" },
                .{ .hf = ".self_attn.v_proj.bias", .hk = ".attn_v.bias" },
                .{ .hf = ".self_attn.o_proj.bias", .hk = ".attn_output.bias" },

                // DeepSeek MLA
                .{ .hf = ".self_attn.q_a_proj.weight", .hk = ".attn_q_a.weight" },
                .{ .hf = ".self_attn.q_b_proj.weight", .hk = ".attn_q_b.weight" },
                .{ .hf = ".self_attn.kv_a_proj_with_mqa.weight", .hk = ".attn_kv_a.weight" },
                .{ .hf = ".self_attn.kv_b_proj.weight", .hk = ".attn_kv_b.weight" },
                .{ .hf = ".self_attn.q_a_layernorm.weight", .hk = ".attn_q_a_norm.weight" },
                .{ .hf = ".self_attn.kv_a_layernorm.weight", .hk = ".attn_kv_a_norm.weight" },

                // MLP Projections
                .{ .hf = ".mlp.gate_proj.weight", .hk = ".mlp_gate.weight" },
                .{ .hf = ".mlp.up_proj.weight", .hk = ".mlp_up.weight" },
                .{ .hf = ".mlp.down_proj.weight", .hk = ".mlp_down.weight" },

                // Norms
                .{ .hf = ".input_layernorm.weight", .hk = ".input_layernorm.weight" },
                .{ .hf = ".post_attention_layernorm.weight", .hk = ".post_attention_layernorm.weight" },
                .{ .hf = ".pre_feedforward_layernorm.weight", .hk = ".pre_ffn_norm.weight" },
                .{ .hf = ".post_feedforward_layernorm.weight", .hk = ".post_ffn_norm.weight" },

                // MoE Gates
                .{ .hf = ".mlp.shared_experts.gate_proj.weight", .hk = ".shared_experts.gate.weight" },
                .{ .hf = ".mlp.shared_experts.up_proj.weight", .hk = ".shared_experts.up.weight" },
                .{ .hf = ".mlp.shared_experts.down_proj.weight", .hk = ".shared_experts.down.weight" },
                .{ .hf = ".mtp_linear.weight", .hk = ".mtp_linear.weight" },

                // Mamba SSM
                .{ .hf = ".mixer.in_proj.weight", .hk = ".ssm_in_proj.weight" },
                .{ .hf = ".mixer.conv1d.weight", .hk = ".ssm_conv1d.weight" },
                .{ .hf = ".mixer.conv1d.bias", .hk = ".ssm_conv1d.bias" },
                .{ .hf = ".mixer.x_proj.weight", .hk = ".ssm_x_proj.weight" },
                .{ .hf = ".mixer.dt_proj.weight", .hk = ".ssm_dt_proj.weight" },
                .{ .hf = ".mixer.dt_proj.bias", .hk = ".ssm_dt_proj.bias" },
                .{ .hf = ".mixer.A_log", .hk = ".ssm_a_log" },
                .{ .hf = ".mixer.D", .hk = ".ssm_d" },
                .{ .hf = ".mixer.out_proj.weight", .hk = ".ssm_out_proj.weight" },
                .{ .hf = ".norm.weight", .hk = ".norm.weight" },
            };

            const is_qwen_moe = std.mem.indexOf(u8, arch, "qwen2_moe") != null or
                std.mem.indexOf(u8, arch, "qwen3_moe") != null or
                std.mem.indexOf(u8, arch, "mixtral") != null or
                std.mem.indexOf(u8, arch, "olmoe") != null;

            if (std.mem.eql(u8, suffix, ".mlp.gate.weight")) {
                const gate_hk = if (is_qwen_moe) ".ffn_gate_inp.weight" else ".moe_gate.weight";
                return try std.fmt.bufPrint(out_buf, "layers.{s}{s}", .{ num, gate_hk });
            }

            for (layer_mappings) |m| {
                if (std.mem.eql(u8, suffix, m.hf)) {
                    return try std.fmt.bufPrint(out_buf, "layers.{s}{s}", .{ num, m.hk });
                }
            }

            // MoE routed experts: .mlp.experts.<E>.gate_proj.weight etc.
            if (std.mem.startsWith(u8, suffix, ".mlp.experts.")) {
                const rest_exp = suffix[13..];
                var exp_num: []const u8 = undefined;
                var exp_suffix: []const u8 = undefined;
                if (extractLayerNum(rest_exp, &exp_num, &exp_suffix)) {
                    if (std.mem.eql(u8, exp_suffix, ".gate_proj.weight")) {
                        return try std.fmt.bufPrint(out_buf, "layers.{s}.experts.{s}.gate.weight", .{ num, exp_num });
                    } else if (std.mem.eql(u8, exp_suffix, ".up_proj.weight")) {
                        return try std.fmt.bufPrint(out_buf, "layers.{s}.experts.{s}.up.weight", .{ num, exp_num });
                    } else if (std.mem.eql(u8, exp_suffix, ".down_proj.weight")) {
                        return try std.fmt.bufPrint(out_buf, "layers.{s}.experts.{s}.down.weight", .{ num, exp_num });
                    }
                }
            }
        }
    }

    // Default fallback: copy unchanged
    if (out_buf.len < name.len) return error.BufferTooSmall;
    @memcpy(out_buf[0..name.len], name);
    return out_buf[0..name.len];
}

/// High-performance HK Canonical tensor name mapping back to Hugging Face format
pub fn mapTensorNameToHf(name: []const u8, arch: []const u8, out_buf: []u8) ![]const u8 {
    _ = arch;
    if (std.mem.eql(u8, name, "embed_tokens.weight")) {
        const target = "model.embed_tokens.weight";
        if (out_buf.len < target.len) return error.BufferTooSmall;
        @memcpy(out_buf[0..target.len], target);
        return out_buf[0..target.len];
    }
    if (std.mem.eql(u8, name, "norm.weight")) {
        const target = "model.norm.weight";
        if (out_buf.len < target.len) return error.BufferTooSmall;
        @memcpy(out_buf[0..target.len], target);
        return out_buf[0..target.len];
    }
    if (std.mem.eql(u8, name, "lm_head.weight")) {
        const target = "lm_head.weight";
        if (out_buf.len < target.len) return error.BufferTooSmall;
        @memcpy(out_buf[0..target.len], target);
        return out_buf[0..target.len];
    }

    if (std.mem.startsWith(u8, name, "layers.")) {
        var num: []const u8 = undefined;
        var suffix: []const u8 = undefined;
        if (extractLayerNum(name[7..], &num, &suffix)) {
            const mapping_entry = struct { hk: []const u8, hf: []const u8 };
            const layer_reverse = [_]mapping_entry{
                .{ .hk = ".attn_q.weight", .hf = ".self_attn.q_proj.weight" },
                .{ .hk = ".attn_k.weight", .hf = ".self_attn.k_proj.weight" },
                .{ .hk = ".attn_v.weight", .hf = ".self_attn.v_proj.weight" },
                .{ .hk = ".attn_output.weight", .hf = ".self_attn.o_proj.weight" },
                .{ .hk = ".attn_q.bias", .hf = ".self_attn.q_proj.bias" },
                .{ .hk = ".attn_k.bias", .hf = ".self_attn.k_proj.bias" },
                .{ .hk = ".attn_v.bias", .hf = ".self_attn.v_proj.bias" },
                .{ .hk = ".attn_output.bias", .hf = ".self_attn.o_proj.bias" },

                .{ .hk = ".attn_q_a.weight", .hf = ".self_attn.q_a_proj.weight" },
                .{ .hk = ".attn_q_b.weight", .hf = ".self_attn.q_b_proj.weight" },
                .{ .hk = ".attn_kv_a.weight", .hf = ".self_attn.kv_a_proj_with_mqa.weight" },
                .{ .hk = ".attn_kv_b.weight", .hf = ".self_attn.kv_b_proj.weight" },
                .{ .hk = ".attn_q_a_norm.weight", .hf = ".self_attn.q_a_layernorm.weight" },
                .{ .hk = ".attn_kv_a_norm.weight", .hf = ".self_attn.kv_a_layernorm.weight" },

                .{ .hk = ".mlp_gate.weight", .hf = ".mlp.gate_proj.weight" },
                .{ .hk = ".mlp_up.weight", .hf = ".mlp.up_proj.weight" },
                .{ .hk = ".mlp_down.weight", .hf = ".mlp.down_proj.weight" },

                .{ .hk = ".input_layernorm.weight", .hf = ".input_layernorm.weight" },
                .{ .hk = ".post_attention_layernorm.weight", .hf = ".post_attention_layernorm.weight" },
                .{ .hk = ".pre_ffn_norm.weight", .hf = ".pre_feedforward_layernorm.weight" },
                .{ .hk = ".post_ffn_norm.weight", .hf = ".post_feedforward_layernorm.weight" },

                .{ .hk = ".moe_gate.weight", .hf = ".mlp.gate.weight" },
                .{ .hk = ".shared_experts.gate.weight", .hf = ".mlp.shared_experts.gate_proj.weight" },
                .{ .hk = ".shared_experts.up.weight", .hf = ".mlp.shared_experts.up_proj.weight" },
                .{ .hk = ".shared_experts.down.weight", .hf = ".mlp.shared_experts.down_proj.weight" },
                .{ .hk = ".mtp_linear.weight", .hf = ".mtp_linear.weight" },
            };

            for (layer_reverse) |m| {
                if (std.mem.eql(u8, suffix, m.hk)) {
                    return try std.fmt.bufPrint(out_buf, "model.layers.{s}{s}", .{ num, m.hf });
                }
            }

            if (std.mem.startsWith(u8, suffix, ".experts.")) {
                const rest_exp = suffix[9..];
                var exp_num: []const u8 = undefined;
                var exp_suffix: []const u8 = undefined;
                if (extractLayerNum(rest_exp, &exp_num, &exp_suffix)) {
                    if (std.mem.eql(u8, exp_suffix, ".gate.weight")) {
                        return try std.fmt.bufPrint(out_buf, "model.layers.{s}.mlp.experts.{s}.gate_proj.weight", .{ num, exp_num });
                    } else if (std.mem.eql(u8, exp_suffix, ".up.weight")) {
                        return try std.fmt.bufPrint(out_buf, "model.layers.{s}.mlp.experts.{s}.up_proj.weight", .{ num, exp_num });
                    } else if (std.mem.eql(u8, exp_suffix, ".down.weight")) {
                        return try std.fmt.bufPrint(out_buf, "model.layers.{s}.mlp.experts.{s}.down_proj.weight", .{ num, exp_num });
                    }
                }
            }
        }
    }

    if (out_buf.len < name.len) return error.BufferTooSmall;
    @memcpy(out_buf[0..name.len], name);
    return out_buf[0..name.len];
}
