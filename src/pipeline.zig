const std = @import("std");

pub const StageModality = enum(c_int) {
    audio_transcription = 0,
    vision_ocr = 1,
    token_analysis = 2,
    classification = 3,
    text_generation = 4,
    custom = 5,
};

pub const PipelineStageDescriptor = struct {
    name: []const u8,
    modality: StageModality,
    model_path: []const u8,
    max_tokens: usize = 128,
    temperature: f32 = 0.7,
};

pub const PipelineContext = struct {
    allocator: std.mem.Allocator,
    total_stages: usize = 0,
    tokens_processed: usize = 0,
    execution_time_ms: f64 = 0.0,

    pub fn init(allocator: std.mem.Allocator) PipelineContext {
        return .{
            .allocator = allocator,
        };
    }
};

/// High-level native pipeline executor state
pub const NativePipelineEngine = struct {
    allocator: std.mem.Allocator,
    stages: std.ArrayList(PipelineStageDescriptor),

    pub fn init(allocator: std.mem.Allocator) NativePipelineEngine {
        return .{
            .allocator = allocator,
            .stages = std.ArrayList(PipelineStageDescriptor).init(allocator),
        };
    }

    pub fn deinit(self: *NativePipelineEngine) void {
        self.stages.deinit();
    }

    pub fn addStage(self: *NativePipelineEngine, stage: PipelineStageDescriptor) !void {
        try self.stages.append(stage);
    }
};
