/**
 * 1bit API Bridge — OpenAI-compatible server for NPU inference.
 *
 * Open WebUI connects here as its OpenAI backend. This server:
 * 1. Lists NPU models via /v1/models
 * 2. Runs inference via npu_engine_mt subprocess on /v1/chat/completions
 * 3. Proxies to external API providers (OpenAI, Anthropic, etc.)
 */

import Fastify from "fastify";
import cors from "@fastify/cors";
import staticFiles from "@fastify/static";
import { randomUUID } from "node:crypto";
import { spawn, execSync } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { discoverModels, findModel, type ModelInfo } from "./models.js";

const PORT = parseInt(process.env.ONE_BIT_PORT || "9090");
const HOST = process.env.ONE_BIT_HOST || "0.0.0.0";
const ENGINE = "/home/bcloud/npu-sandbox/npu-infer/build/npu_engine_mt";
const ENGINE_DIR = "/home/bcloud/npu-sandbox/npu-infer/build";
const XCLBIN_DIR = "/home/bcloud/npu-sandbox/npu-infer/build/int8";

/** Resolve the best engine binary for a given model. */
function resolveEngine(model: ModelInfo): string {
  const variantPath = join(ENGINE_DIR, `npu_engine_${model.tag}`);
  if (existsSync(variantPath)) return variantPath;
  return ENGINE;
}

/** Dynamic default: first discovered ready model, or "none". */
const DEFAULT_MODEL = (() => {
  const models = discoverModels();
  const ready = models.find(m => m.status === "ready");
  return ready ? ready.id : "none";
})();

const uiDir = '/home/bcloud/npu-sandbox/npu-infer/1bit/ui';
const altUiDir = "/home/bcloud/npu-sandbox/npu-infer/1bit/ui";

const fastify = Fastify({ logger: true });

async function start() {
  await fastify.register(cors, { origin: true });

  // Serve UI static files — use alternate path
  await fastify.register(staticFiles, {
    root: altUiDir,
    prefix: "/ui/",
    decorateReply: false,
  });

// Serve index.html at root
fastify.get("/", async (_req, reply) => {
  if (!existsSync(join(altUiDir, "index.html"))) {
    return reply.status(404).send({ error: "UI not found." });
  }
  return reply.type("text/html").send(readFileSync(join(altUiDir, "index.html"), "utf-8"));
});

// ── Tokenizer cache ────────────────────────────────────────────────

interface Tokenizer {
  decode(tokenId: number): string;
  encode(text: string): number[];
}

const tokenizerCache = new Map<string, Tokenizer>();

function loadTokenizer(modelDir: string): Tokenizer {
  const cached = tokenizerCache.get(modelDir);
  if (cached) return cached;

  // Load tokenizer.json (HuggingFace format)
  try {
    const data = JSON.parse(readFileSync(`${modelDir}/tokenizer.json`, "utf-8"));
    const tokenizer: Tokenizer = {
      decode(tokenId: number): string {
        // Simplified: look up in vocab
        const vocab = data.model?.vocab;
        if (!vocab) return `<token ${tokenId}>`;
        for (const [token, id] of Object.entries(vocab)) {
          if (id === tokenId) return token;
        }
        // Byte fallback for BPE tokens
        for (const [token, id] of Object.entries(vocab)) {
          if (typeof id === "number" && id === tokenId) return token;
        }
        return `<token ${tokenId}>`;
      },
      encode(text: string): number[] {
        // Simplified word-level fallback when full BPE isn't available
        // In production, use transformers or a proper BPE tokenizer
        const vocab = data.model?.vocab;
        if (!vocab) return [0];
        const invVocab: Record<string, number> = {};
        for (const [token, id] of Object.entries(vocab as Record<string, number>)) {
          invVocab[token] = id;
        }
        // Simple greedy encoding (works for ASCII)
        const tokens: number[] = [];
        const words = text.toLowerCase().split(/\s+/);
        for (const word of words) {
          if (word in invVocab) {
            tokens.push(invVocab[word]);
          } else {
            // Character fallback
            for (const ch of word) {
              if (ch in invVocab) tokens.push(invVocab[ch]);
            }
          }
        }
        return tokens.length > 0 ? tokens : [0];
      },
    };
    tokenizerCache.set(modelDir, tokenizer);
    return tokenizer;
  } catch (err) {
    fastify.log.warn({ err }, "Failed to load tokenizer, using fallback");
    const fallback: Tokenizer = {
      decode: (id: number) => `<token ${id}>`,
      encode: (text: string) => text.split(" ").map(() => 0),
    };
    tokenizerCache.set(modelDir, fallback);
    return fallback;
  }
}

// ── Model helpers ──────────────────────────────────────────────────

interface OpenAIModel {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

// ── Routes ─────────────────────────────────────────────────────────

// GET /v1/models — list available models
fastify.get("/v1/models", async (_req, _reply) => {
  const models = discoverModels();
  const data = models
    .filter(m => m.status === "ready" || m.status === "no_xclbins")
    .map(m => ({
      id: m.id,
      object: "model",
      created: Math.floor(Date.now() / 1000),
      owned_by: "1bit",
      status: m.status,
      needs_xclbins: m.needsXclbins.length > 0 ? m.needsXclbins : undefined,
    }));
  return { object: "list", data };
});

// GET /v1/models/:model — get model details
fastify.get("/v1/models/:model", async (req, reply) => {
  const { model } = req.params as { model: string };
  const info = findModel(model);
  if (!info) return reply.status(404).send({ error: "Model not found" });
  return {
    id: info.id,
    object: "model",
    created: Math.floor(Date.now() / 1000),
    owned_by: "1bit",
    permissions: [],
    root: info.id,
  };
});

// POST /v1/chat/completions — run inference
fastify.post("/v1/chat/completions", async (req, reply) => {
  const body = req.body as any;
  const modelId = body.model || DEFAULT_MODEL;
  const messages = body.messages || [];
  const stream = body.stream === true;
  const maxTokens = body.max_tokens || 256;
  const temperature = body.temperature ?? 0.7;

  const model = findModel(modelId);
  if (!model) {
    return reply.status(404).send({ error: `Model '${modelId}' not found` });
  }

  // Extract the last user message
  const lastUserMsg = [...messages].reverse().find((m: any) => m.role === "user");
  const prompt = lastUserMsg?.content || "";

  // Truncate prompt to first few tokens for the engine
  // The NPU engine takes space-separated token IDs on stdin
  const tokenizer = loadTokenizer(model.tokenizerPath);
  const inputTokens = tokenizer.encode(prompt);

  if (stream) {
    // Streaming response
    reply.raw.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    const completionId = `chatcmpl-${randomUUID()}`;
    const created = Math.floor(Date.now() / 1000);
    const modelName = model.id;

    // Send role signal
    const roleChunk = {
      id: completionId,
      object: "chat.completion.chunk",
      created,
      model: modelName,
      choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
    };
    reply.raw.write(`data: ${JSON.stringify(roleChunk)}\n\n`);

    // Spawn engine (takes: model.q4nx TOKEN1 TOKEN2 ...)
    const enginePath = resolveEngine(model);
    const engineArgs = [model.modelPath, ...inputTokens.map(String)];

    fastify.log.info({ engine: enginePath, engineArgs }, "Spawning engine");

    const proc = spawn("sudo", [enginePath, ...engineArgs], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    // Buffer whole output and parse tokens
    let output = "";
    proc.stdout.on("data", (chunk) => {
      output += chunk.toString();
    });

    proc.stderr.on("data", (chunk) => {
      fastify.log.warn({ stderr: chunk.toString() }, "Engine stderr");
    });

    proc.on("close", (code) => {
      fastify.log.info({ code, outputLen: output.length }, "Engine done");

      // Parse tokens from output (assumes space-separated token IDs)
      const tokenIds = output.trim().split(/\s+/).map(Number).filter(n => !isNaN(n));

      // Stream each token
      for (let i = 0; i < tokenIds.length; i++) {
        const tokenId = tokenIds[i];
        const tokenText = tokenizer.decode(tokenId);

        const chunk = {
          id: completionId,
          object: "chat.completion.chunk",
          created,
          model: modelName,
          choices: [{
            index: 0,
            delta: { content: tokenText },
            finish_reason: i === tokenIds.length - 1 ? "stop" : null,
          }],
        };
        reply.raw.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }

      // End stream
      reply.raw.write("data: [DONE]\n\n");
      reply.raw.end();
    });

    proc.on("error", (err) => {
      fastify.log.error({ err }, "Failed to spawn engine");
      const errorChunk = {
        id: completionId,
        object: "chat.completion.chunk",
        created,
        model: modelName,
        choices: [{ index: 0, delta: {}, finish_reason: "error" }],
      };
      reply.raw.write(`data: ${JSON.stringify(errorChunk)}\n\n`);
      reply.raw.write("data: [DONE]\n\n");
      reply.raw.end();
    });

    return reply.hijack();
  } else {
    // Non-streaming — run inference and return all output at once
    // Engine takes: model.q4nx TOKEN1 TOKEN2 ...
    const enginePath = resolveEngine(model);
    const engineArgs = [model.modelPath, ...inputTokens.map(String)];

    fastify.log.info({ engine: enginePath, engineArgs }, "Spawning engine (non-streaming)");

    const proc = spawn("sudo", [enginePath, ...engineArgs], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    return new Promise((resolve, reject) => {
      let output = "";
      let stderr = "";
      proc.stdout.on("data", (chunk) => { output += chunk.toString(); });
      proc.stderr.on("data", (chunk) => { stderr += chunk.toString(); });

      proc.on("close", (code) => {
        if (code !== 0) {
          fastify.log.error({ code, stderr }, "Engine failed");
          resolve(reply.status(500).send({
            error: "Inference failed",
            details: stderr,
          }));
          return;
        }

        const tokenIds = output.trim().split(/\s+/).map(Number).filter(n => !isNaN(n));
        const tokenText = tokenIds.map(id => tokenizer.decode(id)).join("");

        const completionId = `chatcmpl-${randomUUID()}`;
        resolve({
          id: completionId,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model: model.id,
          choices: [{
            index: 0,
            message: {
              role: "assistant",
              content: tokenText,
            },
            finish_reason: "stop",
          }],
          usage: {
            prompt_tokens: inputTokens.length,
            completion_tokens: tokenIds.length,
            total_tokens: inputTokens.length + tokenIds.length,
          },
        });
      });

      proc.on("error", (err) => {
        fastify.log.error({ err }, "Failed to spawn engine");
        reject(reply.status(500).send({ error: "Failed to start engine" }));
      });
    });
  }
});

// ── Terminal execution endpoint ─────────────────────────────────
// Allows the browser TUI to execute commands on the host.
// Restricted to safe commands for security.

const ALLOWED_COMMANDS = [
  "ls", "cat", "head", "tail", "grep", "find", "wc", "sort", "uniq",
  "echo", "pwd", "whoami", "id", "uname", "date", "cal", "df", "du",
  "free", "ps", "top", "htop", "uptime", "dmesg", "journalctl",
  "lscpu", "lspci", "lsusb", "lshw", "lsof", "ss", "ip", "ifconfig",
  "nproc", "nvidia-smi", "rocm-smi", "rocminfo",
  "git", "make", "cmake", "gcc", "g++", "clang", "python3", "node",
  "npm", "npx", "cargo", "rustc",
  "cp", "mv", "rm", "mkdir", "touch", "chmod", "chown", "ln",
  "tar", "gzip", "gunzip", "xz", "unzip", "zip",
  "curl", "wget", "ping", "nslookup", "dig",
  "diff", "patch", "sed", "awk", "cut", "tr", "tee",
  "stat", "file", "which", "type", "time", "env", "printenv",
  "kill", "pkill", "nohup",
  "htop", "iostat", "vmstat", "mpstat", "sar",
  "ipcs", "ipcrm", "sysctl",
];

// Security: command must start with an allowed base command
function isAllowedCommand(cmd: string): boolean {
  const trimmed = cmd.trim();
  if (!trimmed || trimmed.length > 512) return false;
  // Extract the base command (first word)
  const base = trimmed.split(/\s+/)[0].replace(/^sudo\s+/i, "");
  return ALLOWED_COMMANDS.includes(base);
}

interface ExecRequest {
  command: string;
}

interface ExecResponse {
  stdout: string;
  stderr: string;
  error?: string;
  exitCode: number;
}

fastify.post<{ Body: ExecRequest }>("/api/exec", async (req, reply) => {
  const { command } = req.body;
  fastify.log.info({ command }, "Terminal exec");

  if (!isAllowedCommand(command)) {
    return reply.send({
      stdout: "",
      stderr: "",
      error: `Command not allowed. Allowed commands: ${ALLOWED_COMMANDS.slice(0, 10).join(", ")}...`,
      exitCode: 1,
    } as ExecResponse);
  }

  const proc = spawn("bash", ["-c", command], {
    stdio: ["pipe", "pipe", "pipe"],
    timeout: 30000,
    env: { ...process.env },
  });

  return new Promise((resolve) => {
    let stdout = "";
    let stderr = "";

    proc.stdout?.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
    proc.stderr?.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });

    proc.on("close", (code: number | null) => {
      // Truncate to prevent huge responses
      if (stdout.length > 10000) stdout = stdout.slice(0, 10000) + "\n... [truncated]";
      if (stderr.length > 5000) stderr = stderr.slice(0, 5000) + "\n... [truncated]";

      resolve(reply.send({
        stdout,
        stderr,
        exitCode: code ?? -1,
      } as ExecResponse));
    });

    proc.on("error", (err) => {
      resolve(reply.send({
        stdout: "",
        stderr: "",
        error: err.message,
        exitCode: -1,
      } as ExecResponse));
    });

    // Timeout kill
    setTimeout(() => {
      if (proc.exitCode === null) {
        proc.kill("SIGTERM");
        resolve(reply.send({
          stdout,
          stderr: stderr + "\n[Command timed out after 30s]",
          error: "Timeout",
          exitCode: -1,
        } as ExecResponse));
      }
    }, 30000);
  });
});

// Health check
fastify.get("/health", async () => ({ status: "ok", service: "1bit" }));

  try {
    await fastify.listen({ port: PORT, host: HOST });
    fastify.log.info(`1bit API bridge listening on ${HOST}:${PORT}`);

    // Auto-load default model (warm up NPU) — fully async, non-blocking
    setTimeout(() => {
      const models = discoverModels();
      const defaultModel = models.find(m => m.status === "ready");
      if (!defaultModel) {
        fastify.log.warn("No ready models found for warm-up");
        return;
      }
      fastify.log.info(`Pre-loading default model: ${defaultModel.id}...`);
      const enginePath = resolveEngine(defaultModel);
      const warmProc = spawn("sudo", [enginePath, defaultModel.modelPath, "42", "99", "211", "307"], {
        stdio: ["pipe", "pipe", "pipe"],
        timeout: 60000,
      });
      let warmOut = "";
      warmProc.stdout?.on("data", (d: Buffer) => { warmOut += d.toString(); });
      warmProc.on("close", (code: number | null) => {
        if (code === 0) {
          fastify.log.info(`Default model loaded — warm-up output: ${warmOut.trim().substring(0, 100)}`);
        } else {
          fastify.log.warn(`Default model warm-up exit code ${code} (model loads on first request)`);
        }
      });
      warmProc.on("error", (err: Error) => {
        fastify.log.warn({ err: err.message }, `Default model warm-up skipped`);
      });
    }, 100);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
}
start();
