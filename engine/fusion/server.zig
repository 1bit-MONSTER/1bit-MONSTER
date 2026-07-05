//! Unified HTTP server for NPU+GPU fused inference.
//! Serves the OpenAI-compatible /v1/chat/completions API through the FusedEngine,
//! routing requests to NPU, GPU, or both based on the dispatch policy.
//!
//! @section Fused Engine
const std = @import("std");

const eng = @import("engine.zig");
const dispatcher = @import("dispatcher.zig");

pub const FusedEngine = eng.FusedEngine;
pub const Dispatcher = dispatcher.Dispatcher;
pub const DispatchPolicy = dispatcher.DispatchPolicy;
pub const KvPagePool = eng.KvPagePool;
pub const Scheduler = eng.Scheduler;
pub const Request = eng.Request;
pub const RequestState = eng.RequestState;
pub const GenerationParams = eng.GenerationParams;

const log = std.log.scoped(.fusion_server);

/// HTTP connection wrapper (similar to the GPU engine's http.zig).
pub const Connection = struct {
    fd: std.posix.fd_t,
    buf_reader: std.io.BufferedReader(4096, std.fs.File.Reader),
    conn_writer: std.fs.File.Writer,

    pub fn init(fd: std.posix.fd_t) Connection {
        const file = std.fs.File{ .handle = fd };
        return .{
            .fd = fd,
            .buf_reader = std.io.bufferedReader(file.reader()),
            .conn_writer = file.writer(),
        };
    }

    pub fn close(self: *Connection) void {
        std.posix.close(self.fd);
        self.* = undefined;
    }

    pub fn reader(self: *Connection) std.io.BufferedReader(4096, std.fs.File.Reader).Reader {
        return self.buf_reader.reader();
    }

    pub fn writer(self: *Connection) std.fs.File.Writer {
        return self.conn_writer;
    }
};

/// Response buffer for building HTTP responses.
pub const ResponseBuilder = struct {
    buf: std.ArrayList(u8),

    pub fn init(allocator: std.mem.Allocator) ResponseBuilder {
        return .{ .buf = std.ArrayList(u8).init(allocator) };
    }

    pub fn deinit(self: *ResponseBuilder) void {
        self.buf.deinit();
    }

    pub fn reset(self: *ResponseBuilder) void {
        self.buf.clearRetainingCapacity();
    }

    pub fn writeStatus(self: *ResponseBuilder, status: u16, status_text: []const u8) !void {
        try self.buf.writer().print("HTTP/1.1 {d} {s}\r\n", .{ status, status_text });
    }

    pub fn writeHeader(self: *ResponseBuilder, name: []const u8, value: []const u8) !void {
        try self.buf.writer().print("{s}: {s}\r\n", .{ name, value });
    }

    pub fn writeBody(self: *ResponseBuilder, body: []const u8) !void {
        try self.writeHeader("Content-Length", &[_]u8{});
        const len_str = try std.fmt.allocPrint(self.buf.allocator, "{d}", .{body.len});
        defer self.buf.allocator.free(len_str);
        // Re-write Content-Length with actual value by replacing the header
        // For simplicity: append body after headers
        try self.buf.appendSlice("\r\n");
        try self.buf.appendSlice(body);
    }

    pub fn finish(self: *ResponseBuilder) ![]const u8 {
        // Finalize headers if not already done
        // Simple approach: return the buffer contents
        return self.buf.items;
    }

    pub fn sendJson(self: *ResponseBuilder, conn: *Connection, status: u16, json_body: []const u8) !void {
        try self.writeStatus(status, if (status == 200) "OK" else if (status == 400) "Bad Request" else if (status == 404) "Not Found" else if (status == 500) "Internal Server Error" else "Unknown");
        try self.writeHeader("Content-Type", "application/json");
        try self.writeHeader("Access-Control-Allow-Origin", "*");
        try self.writeHeader("Content-Length", &[_]u8{});
        const len_str = try std.fmt.allocPrint(self.buf.allocator, "{d}", .{json_body.len});
        defer self.buf.allocator.free(len_str);
        try self.writeHeader("Content-Length", len_str);
        try self.buf.appendSlice("\r\n");
        try self.buf.appendSlice(json_body);
        try conn.writer().writeAll(self.buf.items);
        self.reset();
    }
};

/// Fused engine server configuration.
pub const ServerConfig = struct {
    port: u16 = 8080,
    max_parallel: u32 = 4,
    total_kv_pages: u32 = 1024,
    dispatch_policy: DispatchPolicy = .auto,
    model_path: []const u8 = "",
    xclbin_dir: []const u8 = "",
    model_tag: []const u8 = "",
};

/// Handle one HTTP request on the fused engine server.
pub fn handleRequest(
    conn: *Connection,
    engine: *FusedEngine,
    allocator: std.mem.Allocator,
) !void {
    var resp = ResponseBuilder.init(allocator);
    defer resp.deinit();

    // Read request line
    const reader = conn.reader();
    const request_line = reader.readUntilDelimiterAlloc(allocator, '\n', 4096) catch |err| {
        log.warn("Failed to read request line: {s}", .{@errorName(err)});
        try resp.sendJson(conn, 400, "{\"error\":\"bad request\"}");
        return;
    };
    defer allocator.free(request_line);
    const req = std.mem.trim(u8, request_line, "\r\n ");

    // Parse method and path
    var parts = std.mem.splitScalar(u8, req, ' ');
    const method = parts.next() orelse {
        try resp.sendJson(conn, 400, "{\"error\":\"bad request\"}");
        return;
    };
    const path = parts.next() orelse {
        try resp.sendJson(conn, 400, "{\"error\":\"bad request\"}");
        return;
    };

    _ = method;

    // Route to handler
    if (std.mem.eql(u8, path, "/health")) {
        try resp.sendJson(conn, 200, "{\"status\":\"ok\"}");
    } else if (std.mem.eql(u8, path, "/v1/chat/completions")) {
        try handleChatCompletions(conn, engine, &resp, reader, allocator);
    } else if (std.mem.eql(u8, path, "/v1/completions")) {
        try handleCompletions(conn, engine, &resp, reader, allocator);
    } else if (std.mem.eql(u8, path, "/v1/models")) {
        try handleModels(conn, engine, &resp, allocator);
    } else {
        try resp.sendJson(conn, 404, "{\"error\":\"not found\"}");
    }
}

fn handleChatCompletions(
    conn: *Connection,
    _: *FusedEngine,
    resp: *ResponseBuilder,
    reader: anytype,
    allocator: std.mem.Allocator,
) !void {
    // engine unused until routing is wired
    // Read headers and body
    skipHeaders(reader, allocator) catch {};

    const body = reader.readUntilDelimiterAlloc(allocator, '\n', 65536) catch |err| {
        log.warn("Failed to read request body: {s}", .{@errorName(err)});
        try resp.sendJson(conn, 400, "{\"error\":\"bad request\"}");
        return;
    };
    defer allocator.free(body);

    // Parse JSON body
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        body,
        .{ .ignore_unknown_fields = true },
    ) catch |err| {
        log.warn("Failed to parse JSON: {s}", .{@errorName(err)});
        try resp.sendJson(conn, 400, "{\"error\":\"invalid json\"}");
        return;
    };
    defer parsed.deinit();

    const root = parsed.value;

    // Extract prompt from messages
    const messages = root.object.get("messages") orelse {
        try resp.sendJson(conn, 400, "{\"error\":\"missing messages\"}");
        return;
    };

    // Simple extraction: take last user message content
    const msg_array = messages.array.items;
    // Extract prompt from last user message
    for (msg_array) |msg| {
        if (msg.object.get("role")) |role| {
            if (std.mem.eql(u8, role.string, "user")) {
                if (msg.object.get("content")) |_| {
                    // prompt found — will wire through tokenizer
                }
            }
        }
    }

    // TODO: Tokenize, run inference, produce response
    // For now, return a placeholder
    try resp.sendJson(conn, 200,
        \\{"id":"chatcmpl-fused","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hello from the fused NPU+GPU engine!"},"finish_reason":"stop"}]}
    );
    // prompt_text will be used once tokenize/route is wired
}

fn handleCompletions(
    conn: *Connection,
    _: *FusedEngine,
    resp: *ResponseBuilder,
    reader: anytype,
    allocator: std.mem.Allocator,
) !void {
    // engine unused until routing is wired
    skipHeaders(reader, allocator) catch {};

    const body = reader.readUntilDelimiterAlloc(allocator, '\n', 65536) catch |err| {
        log.warn("Failed to read request body: {s}", .{@errorName(err)});
        try resp.sendJson(conn, 400, "{\"error\":\"bad request\"}");
        return;
    };
    defer allocator.free(body);
    try resp.sendJson(conn, 200,
        \\{"id":"cmpl-fused","object":"text_completion","choices":[{"text":"Fused NPU+GPU engine active.","index":0,"finish_reason":"stop"}]}
    );
}

fn handleModels(
    conn: *Connection,
    _: *FusedEngine,
    resp: *ResponseBuilder,
    allocator: std.mem.Allocator,
) !void {
    // caps unused until routing is wired
    // engine unused until routing is wired

    const model_list = try std.fmt.allocPrint(allocator,
        \\{{"object":"list","data":[{{"id":"fused-engine","object":"model","created":{d},"owned_by":"1bit.systems","description":"NPU+GPU fused inference engine (NPU INT8 GEMM + GPU attention)"}}]}}
    , .{@as(u64, @intCast(std.time.timestamp()))});
    defer allocator.free(model_list);

    try resp.sendJson(conn, 200, model_list);
}

fn skipHeaders(reader: anytype, allocator: std.mem.Allocator) !void {
    while (true) {
        const line = reader.readUntilDelimiterAlloc(allocator, '\n', 4096) catch return;
        defer allocator.free(line);
        const trimmed = std.mem.trim(u8, line, "\r\n ");
        if (trimmed.len == 0) return;
    }
}

/// Run the fused engine HTTP server.
pub fn runServer(config: ServerConfig, allocator: std.mem.Allocator) !void {
    log.info("Starting fused NPU+GPU server on port {d} (policy: {s})", .{
        config.port, @tagName(config.dispatch_policy),
    });

    // Initialize the fused engine
    var fused = try FusedEngine.init(
        allocator,
        config.model_path,
        config.xclbin_dir,
        config.model_tag,
        config.max_parallel,
        config.total_kv_pages,
        config.dispatch_policy,
    );
    defer fused.deinit();

    // Bind and listen
    const addr = std.posix.sockaddr_in.init(
        std.posix.INADDR_LOOPBACK,
        std.posix.htons(config.port),
    );
    const sock_fd = try std.posix.socket(
        std.posix.AF.INET,
        std.posix.SOCK.STREAM,
        std.posix.IPPROTO.TCP,
    );
    defer std.posix.close(sock_fd);

    const reuse: c_int = 1;
    try std.posix.setsockopt(sock_fd, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, std.mem.asBytes(&reuse));

    try std.posix.bind(sock_fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr)));
    try std.posix.listen(sock_fd, 128);

    log.info("Fused server listening on 0.0.0.0:{d}", .{config.port});

    // Accept loop
    while (true) {
        var client_addr: std.posix.sockaddr = undefined;
        var client_addr_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);
        const client_fd = std.posix.accept(sock_fd, &client_addr, &client_addr_len, 0) catch |err| {
            log.warn("Accept failed: {s}", .{@errorName(err)});
            continue;
        };

        var conn = Connection.init(client_fd);
        handleRequest(&conn, &fused, allocator) catch |err| {
            log.warn("Request failed: {s}", .{@errorName(err)});
        };
        conn.close();
    }
}
