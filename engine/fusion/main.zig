//! Fused NPU+GPU inference engine — entry point.
//! Builds and runs the fused engine with CLI configuration.
//!
//! Usage:
//!   fused-engine --model <model.q4nx> --xclbin-dir <dir> [--port 8080] [--policy auto]
//!   fused-engine --model <model.q4nx> --prompt "Hello" [--max-tokens 128]
//!
//! @section Fused Engine
const std = @import("std");

const server = @import("server.zig");
const dispatcher = @import("dispatcher.zig");

pub const DispatchPolicy = dispatcher.DispatchPolicy;
pub const ServerConfig = server.ServerConfig;

const log = std.log.scoped(.fusion_main);

/// CLI configuration.
pub const Config = struct {
    /// Path to NPU model file (.q4nx).
    model_path: ?[]const u8 = null,
    /// Directory containing xclbin files.
    xclbin_dir: ?[]const u8 = null,
    /// Model tag for xclbin selection.
    model_tag: ?[]const u8 = null,
    /// Server port (default: 8080, or 9090 for chat).
    port: u16 = 8080,
    /// Max concurrent requests.
    max_parallel: u32 = 4,
    /// Total KV cache pages.
    total_kv_pages: u32 = 1024,
    /// Dispatch policy (npu_only, gpu_only, auto, etc.).
    dispatch_policy: DispatchPolicy = .auto,
    /// CLI prompt text (run once, no server).
    prompt: ?[]const u8 = null,
    /// Max decode tokens for CLI mode.
    max_tokens: u32 = 128,
    /// Show help.
    show_help: bool = false,
};

const help_text =
    \\Fused NPU+GPU Inference Engine — 1bit.systems
    \\
    \\Usage:
    \\  fused-engine --model <model.q4nx> [options]
    \\  fused-engine --model <model.q4nx> --prompt "Hello" [--max-tokens 128]
    \\
    \\Options:
    \\  --model <path>         NPU model file (.q4nx)
    \\  --xclbin-dir <dir>     XCLBIN directory (default: ~/npu-sandbox/npu-infer/build/int8)
    \\  --model-tag <tag>      Model tag for xclbin selection
    \\  -p, --port <port>      Server port (default: 8080)
    \\  --parallel <n>         Max concurrent requests (default: 4)
    \\  --kv-pages <n>         Total KV cache pages (default: 1024)
    \\  --policy <policy>      Dispatch policy: auto (default), npu_only, gpu_only,
    \\                         attention_on_npu, ffn_on_npu, qkv_on_npu, layer_by_layer,
    \\                         prefill_npu_decode_gpu
    \\  --prompt <text>        Run one prompt in CLI mode
    \\  -n, --max-tokens <n>   Max generated tokens (default: 128)
    \\  -h, --help             Show this help
    \\
    \\Dispatch Policies:
    \\  auto                  Auto-tuned: FFN/QKV→NPU, Attention→GPU
    \\  npu_only              All layers → NPU (XRT xclbin INT8)
    \\  gpu_only              All layers → GPU (Vulkan/Metal/CUDA)
    \\  attention_on_npu      Attention → NPU edge_attention, FFN → GPU DMMV
    \\  ffn_on_npu            FFN → NPU INT8 GEMM, Attention → GPU flash attn
    \\  layer_by_layer        Alternating layers between NPU and GPU
    \\
;

pub fn parseArgs(args: []const [:0]const u8) !Config {
    var config = Config{};
    var i: usize = 1;

    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            config.show_help = true;
            return config;
        } else if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--xclbin-dir")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.xclbin_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--model-tag")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.model_tag = args[i];
        } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.port = std.fmt.parseInt(u16, args[i], 10) catch return error.InvalidPort;
        } else if (std.mem.eql(u8, arg, "--parallel")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.max_parallel = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidParallel;
        } else if (std.mem.eql(u8, arg, "--kv-pages")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.total_kv_pages = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidKvPages;
        } else if (std.mem.eql(u8, arg, "--policy")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.dispatch_policy = parsePolicy(args[i]) catch return error.InvalidPolicy;
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.prompt = args[i];
        } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.max_tokens = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidMaxTokens;
        } else {
            return error.UnknownArgument;
        }
    }

    return config;
}

fn parsePolicy(name: []const u8) !DispatchPolicy {
    if (std.mem.eql(u8, name, "npu_only")) return .npu_only;
    if (std.mem.eql(u8, name, "gpu_only")) return .gpu_only;
    if (std.mem.eql(u8, name, "attention_on_npu")) return .attention_on_npu;
    if (std.mem.eql(u8, name, "ffn_on_npu")) return .ffn_on_npu;
    if (std.mem.eql(u8, name, "qkv_on_npu")) return .qkv_on_npu;
    if (std.mem.eql(u8, name, "layer_by_layer")) return .layer_by_layer;
    if (std.mem.eql(u8, name, "prefill_npu_decode_gpu")) return .prefill_npu_decode_gpu;
    if (std.mem.eql(u8, name, "auto")) return .auto;
    return error.InvalidPolicy;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const config = Config{
        .model_path = null,
    };

    if (config.show_help) {
        std.debug.print("{s}", .{help_text});
        return;
    }

    log.info("Fused NPU+GPU Engine starting...", .{});

    if (config.prompt) |prompt| {
        // CLI mode: run one prompt, print output, exit
        std.debug.print("CLI mode not yet implemented (prompt: {s})\n", .{prompt});
        std.debug.print("Start the server with: fused-engine --model <path> --port 8080\n", .{});
        return;
    }

    // Server mode
    const model_path = config.model_path orelse {
        std.debug.print("Error: --model is required for server mode\n\n", .{});
        std.debug.print("{s}", .{help_text});
        std.process.exit(1);
    };

    const xclbin_dir = config.xclbin_dir orelse
        "/home/bcloud/npu-sandbox/npu-infer/build/int8";

    var server_config = ServerConfig{
        .port = config.port,
        .max_parallel = config.max_parallel,
        .total_kv_pages = config.total_kv_pages,
        .dispatch_policy = config.dispatch_policy,
        .model_path = model_path,
        .xclbin_dir = xclbin_dir,
    };

    if (config.model_tag) |tag| {
        server_config.model_tag = tag;
    }

    server.runServer(server_config, allocator) catch |err| {
        log.err("Server failed: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
}
