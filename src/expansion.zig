const std = @import("std");

pub const ExpansionDiagnosis = struct {
    needs_expansion: bool,
    needs_vocab_expansion: bool,
    missing_token_count: usize,
    needs_width_expansion: bool,
    suggested_width_ratio: f32,
    diagnostic_loss: f32,
    pass_rate: f32,
};

/// Evaluates whether an architecture expansion is needed for a target domain in native Zig.
pub fn evaluateExpansionNeed(
    diagnostic_loss: f32,
    pass_rate: f32,
    known_vocab_count: usize,
    domain_keyword_count: usize,
    unrecognized_keywords: usize,
    current_intermediate: usize,
    max_growth_ratio: f32,
) ExpansionDiagnosis {
    _ = known_vocab_count;
    _ = domain_keyword_count;

    const needs_vocab = (unrecognized_keywords > 0);
    const error_rate = 1.0 - pass_rate;

    var needs_width = false;
    var suggested_ratio: f32 = 1.0;

    if (error_rate > 0.40 or diagnostic_loss > 4.0) {
        needs_width = true;
        if (error_rate > 0.70) {
            suggested_ratio = 1.50;
        } else {
            suggested_ratio = 1.33;
        }

        // Enforce max growth ratio
        if (suggested_ratio > max_growth_ratio) {
            suggested_ratio = max_growth_ratio;
        }
    }

    _ = current_intermediate;

    return ExpansionDiagnosis{
        .needs_expansion = (needs_vocab or needs_width),
        .needs_vocab_expansion = needs_vocab,
        .missing_token_count = unrecognized_keywords,
        .needs_width_expansion = needs_width,
        .suggested_width_ratio = suggested_ratio,
        .diagnostic_loss = diagnostic_loss,
        .pass_rate = pass_rate,
    };
}
