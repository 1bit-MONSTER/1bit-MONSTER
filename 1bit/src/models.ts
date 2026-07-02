/**
 * Model registry — dynamically discovers Q4NX models from disk.
 * Reads HF config.json for dimensions, derives xclbin tags.
 * No hardcoded model lists.
 */

import { readFileSync, readdirSync, existsSync, openSync, readSync, closeSync } from "node:fs";
import { join } from "node:path";

export interface ModelInfo {
  id: string;            // model ID for OpenAI API (directory name)
  name: string;          // display name
  modelPath: string;     // path to model.q4nx (or dir if no .q4nx)
  tokenizerPath: string; // path to dir containing tokenizer.json
  configPath: string;    // path to config.json
  modelType: string;     // "qwen2", "llama", "gemma4_text", etc.
  tag: string;           // derived tag for xclbin selection
  dimensions: {
    H: number; NC: number; NH: number; NKV: number;
    HD: number; IM: number; NV: number;
  };
  architecture: {
    q_norm: boolean; k_norm: boolean;
    rope_freqs: boolean; lm_head: boolean; gu_split: boolean;
  };
  status: "ready" | "no_xclbins" | "no_model_file" | "no_config";
  needsXclbins: string[];  // list of missing xclbin operation names
}

const MODELS_BASE = "/home/bcloud/.config/flm/models";
const MODELS_BASE_FALLBACK = "/home/bcloud/models";
const XCLBIN_DIR = "/home/bcloud/npu-sandbox/npu-infer/build/int8";

/** HuggingFace config.json fields we consume. */
interface HfConfig {
  hidden_size?: number;
  intermediate_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  vocab_size?: number;
  model_type?: string;
  max_position_embeddings?: number;
  rope_theta?: number;
  rms_norm_eps?: number;
  /** Some models (Qwen) use `num_layers` instead of `num_hidden_layers`. */
  num_layers?: number;
  /** Some models (Gemma) set head_dim explicitly. */
  head_dim?: number;
}

function readConfig(dir: string): HfConfig | null {
  try {
    const raw = readFileSync(join(dir, "config.json"), "utf-8");
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function findQ4NXFile(dir: string): string | null {
  try {
    const entries = readdirSync(dir);
    const q4nx = entries.find(e => e.endsWith(".q4nx"));
    return q4nx ? join(dir, q4nx) : null;
  } catch {
    return null;
  }
}

/**
 * Derive xclbin tag from model type + hidden_size.
 * Known architectures get their existing tags for xclbin compatibility;
 * unknown ones get `${modelType}_${H}`.
 */
function deriveTag(modelType: string, H: number): string {
  const known: Record<string, Record<number, string>> = {
    qwen2: { 1024: "qwen3_0_6b", 2560: "qwen3_vl_4b", 4096: "qwen3_8b" },
    qwen3: { 1024: "qwen3_0_6b", 2560: "qwen3_vl_4b", 4096: "qwen3_8b" },
    llama: { 4096: "llama" },
    gemma2: { 1536: "gemma4_e2b", 2304: "gemma4_e2b" },
    gemma4_text: { 1536: "gemma4_e2b", 2304: "gemma4_e2b" },
  };
  return known[modelType]?.[H] || `${modelType}_${H}`;
}

/**
 * Determine required xclbin ops.
 * Models with intermediate_size > 7168 use separate G/U xclbins (GuSplit).
 * Smaller models use fused GU xclbin.
 */
function requiredOps(H: number, IM: number): string[] {
  const guSplit = (2 * IM > 14336);
  const base = ["QKV", "O", "D"];
  if (guSplit) return [...base, "G", "U"];
  return [...base, "GU"];
}

/**
 * Check which xclbins exist for a given tag and required ops.
 * Tries tag-specific xclbin first, then falls back to generic.
 */
function checkXclbins(tag: string, ops: string[]): { status: "ready" | "no_xclbins"; missing: string[] } {
  const missing: string[] = [];
  for (const op of ops) {
    const specificPath = join(XCLBIN_DIR, `final_i8_${op}_${tag}.xclbin`);
    const genericPath = join(XCLBIN_DIR, `final_i8_${op}.xclbin`);
    if (!existsSync(specificPath) && !existsSync(genericPath)) {
      missing.push(op);
    }
  }
  return { status: missing.length ? "no_xclbins" : "ready", missing };
}

/**
 * Parse Q4NX header to extract architecture info.
 * The header is an 8-byte size prefix followed by a JSON payload.
 */
function parseQ4NXHeader(filePath: string): Record<string, any> | null {
  try {
    const fd = openSync(filePath, "r");
    try {
      const hdrSizeBuf = Buffer.alloc(8);
      readSync(fd, hdrSizeBuf, 0, 8, 0);
      const hdrSize = Number(hdrSizeBuf.readBigUInt64LE(0));
      if (hdrSize <= 0 || hdrSize > 1024 * 1024) return null;

      const headerBuf = Buffer.alloc(hdrSize);
      readSync(fd, headerBuf, 0, hdrSize, 8);
      const header = headerBuf.toString("utf-8").replace(/\x00+$/, "");
      return JSON.parse(header);
    } finally {
      closeSync(fd);
    }
  } catch {
    return null;
  }
}

export function discoverModels(): ModelInfo[] {
  const modelDirs = new Map<string, string>();

  // Collect all model directories from primary and fallback paths
  for (const base of [MODELS_BASE, MODELS_BASE_FALLBACK]) {
    try {
      const entries = readdirSync(base);
      for (const entry of entries) {
        const fullPath = join(base, entry);
        // Skip non-directories
        try {
          if (!existsSync(fullPath)) continue;
        } catch {
          continue;
        }
        if (!modelDirs.has(entry)) modelDirs.set(entry, fullPath);
      }
    } catch {
      // Path may not exist
    }
  }

  const models: ModelInfo[] = [];

  for (const [dirName, dirPath] of modelDirs) {
    const q4nxPath = findQ4NXFile(dirPath);
    const config = readConfig(dirPath);

    // No Q4NX file found
    if (!q4nxPath) {
      models.push({
        id: dirName,
        name: dirName,
        modelPath: dirPath,
        tokenizerPath: dirPath,
        configPath: join(dirPath, "config.json"),
        modelType: config?.model_type || "unknown",
        tag: "unknown",
        dimensions: { H: 0, NC: 0, NH: 0, NKV: 0, HD: 128, IM: 0, NV: 0 },
        architecture: {
          q_norm: false, k_norm: false,
          rope_freqs: false, lm_head: false, gu_split: false,
        },
        status: config ? "no_model_file" : "no_config",
        needsXclbins: [],
      });
      continue;
    }

    // Q4NX found but no config.json
    if (!config) {
      models.push({
        id: dirName,
        name: dirName,
        modelPath: q4nxPath,
        tokenizerPath: dirPath,
        configPath: join(dirPath, "config.json"),
        modelType: "unknown",
        tag: "unknown",
        dimensions: { H: 0, NC: 0, NH: 0, NKV: 0, HD: 128, IM: 0, NV: 0 },
        architecture: {
          q_norm: false, k_norm: false,
          rope_freqs: false, lm_head: false, gu_split: false,
        },
        status: "no_config",
        needsXclbins: [],
      });
      continue;
    }

    // Both Q4NX and config.json exist — derive dimensions
    const H = config.hidden_size ?? 0;
    const IM = config.intermediate_size ?? 0;
    const NC = config.num_hidden_layers ?? config.num_layers ?? 0;
    const NH = config.num_attention_heads ?? 0;
    const NKV = config.num_key_value_heads ?? NH;
    const HD = config.head_dim ?? (NH > 0 ? Math.floor(H / NH) : 128);
    const NV = config.vocab_size ?? 0;
    const modelType = config.model_type ?? "unknown";

    const tag = deriveTag(modelType, H);
    const ops = requiredOps(H, IM);
    const { status, missing } = checkXclbins(tag, ops);

    // Architecture flags based on model type
    const hasQNorm = modelType.startsWith("qwen") || modelType.startsWith("gemma");
    const hasKNorm = modelType.startsWith("qwen") || modelType.startsWith("gemma");
    const hasRope = (modelType === "llama" || modelType.startsWith("gemma") ||
                     modelType === "mistral" || modelType.startsWith("qwen"));
    const guSplit = (2 * IM > 14336);

    models.push({
      id: dirName,
      name: dirName,
      modelPath: q4nxPath,
      tokenizerPath: dirPath,
      configPath: join(dirPath, "config.json"),
      modelType,
      tag,
      dimensions: { H, NC, NH, NKV, HD, IM, NV },
      architecture: {
        q_norm: hasQNorm, k_norm: hasKNorm,
        rope_freqs: hasRope, lm_head: false, gu_split: guSplit,
      },
      status,
      needsXclbins: missing,
    });
  }

  return models;
}

export function findModel(id: string): ModelInfo | undefined {
  return discoverModels().find(m => m.id === id);
}

export function getModels(): ModelInfo[] {
  return discoverModels();
}
