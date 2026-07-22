//! Voice-cloning engine. Port of jarvis/voice/engine.py + jarvis/voice/codec.py.
//!
//! Only the **decode** path is ported (encode/train aren't used by the live
//! server -- see jarvis/voice/train.py, dataset.py, record.py, which stay
//! Python dev tooling per the Rust-port plan). `VoicePack::synthesize()`'s
//! text->token mapping is copied verbatim from the Python
//! `VoicePack.synthesize()` docstring's own description: an explicitly
//! labeled Phase-1 placeholder (character ordinals -> fake codec tokens),
//! not a real acoustic model. There's no "golden" reference output to
//! verify against even in the Python original, and no `.voice` pack exists
//! anywhere on this box yet (`voice/record.py` has never been run) --
//! this is implemented faithfully against the architecture in codec.py but
//! is unverified end-to-end against real trained weights, unlike every
//! other module in this port.

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct CodecConfig {
    pub sample_rate: usize,
    pub hop_length: usize,
    pub latent_dim: usize,
    pub n_codebooks: usize,
    pub codebook_dim: usize,
    pub decoder_strides: [usize; 4],
    pub hidden_channels: usize,
    pub input_channels: usize,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            hop_length: 320,
            latent_dim: 128,
            n_codebooks: 4,
            codebook_dim: 128,
            decoder_strides: [8, 5, 4, 2],
            hidden_channels: 64,
            input_channels: 1,
        }
    }
}

struct DecoderBlock {
    weight: Tensor, // (in_ch, out_ch, kernel) -- PyTorch ConvTranspose1d layout
    bias: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    stride: usize,
    padding: usize,
}

pub struct CodecDecoder {
    codebooks: Vec<Tensor>, // n_codebooks x (codebook_size, codebook_dim)
    codebook_proj: Option<(Tensor, Tensor)>, // (weight, bias), only if codebook_dim != latent_dim
    blocks: Vec<DecoderBlock>,
    proj_weight: Tensor,
    proj_bias: Tensor,
    cfg: CodecConfig,
    device: Device,
}

fn elu(x: &Tensor) -> candle_core::Result<Tensor> {
    // nn.ELU() default alpha=1.0: x if x>0 else exp(x)-1
    let pos = x.relu()?;
    let neg_mask = x.le(0f64)?.to_dtype(x.dtype())?;
    let expm1 = (x.exp()? - 1.0)?;
    let neg = (expm1 * neg_mask)?;
    pos + neg
}

fn group_norm_1group(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    // num_groups=1 over (B, C, T) -- normalize over the whole (C,T) per batch
    // element, matching nn.GroupNorm(1, out_ch).
    let (b, c, t) = x.dims3()?;
    let flat = x.reshape((b, c * t))?;
    let mean = flat.mean_keepdim(1)?;
    let centered = flat.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(1)?;
    let std = (var + eps)?.sqrt()?;
    let normed = centered.broadcast_div(&std)?.reshape((b, c, t))?;
    let w = weight.reshape((1, c, 1))?;
    let bi = bias.reshape((1, c, 1))?;
    normed.broadcast_mul(&w)?.broadcast_add(&bi)
}

impl CodecDecoder {
    pub fn load(path: &Path, device: &Device) -> Result<Self, String> {
        let cfg = CodecConfig::default();
        let tensors: Vec<(String, Tensor)> = candle_core::pickle::read_all(path).map_err(|e| e.to_string())?;
        let map: HashMap<String, Tensor> = tensors.into_iter().collect();
        let get = |name: &str| -> Result<Tensor, String> {
            map.get(name).cloned().ok_or_else(|| format!("missing tensor: {name}")).and_then(|t| t.to_dtype(DType::F32).map_err(|e| e.to_string()))
        };

        let mut codebooks = Vec::with_capacity(cfg.n_codebooks);
        for i in 0..cfg.n_codebooks {
            codebooks.push(get(&format!("quantizer.codebooks.{i}.weight"))?.to_device(device).map_err(|e| e.to_string())?);
        }

        let codebook_proj = if cfg.codebook_dim != cfg.latent_dim {
            Some((
                get("codebook_proj.weight")?.to_device(device).map_err(|e| e.to_string())?,
                get("codebook_proj.bias")?.to_device(device).map_err(|e| e.to_string())?,
            ))
        } else {
            None
        };

        // Mirrors Decoder.__init__'s channel schedule exactly.
        let strides = cfg.decoder_strides;
        let mut channels = vec![cfg.latent_dim];
        for i in 0..strides.len() {
            channels.push((cfg.hidden_channels * (1 << (strides.len() - 1 - i))).min(512));
        }
        let mut blocks = Vec::with_capacity(strides.len());
        for i in 0..strides.len() {
            let stride = strides[i];
            let padding = stride / 2;
            blocks.push(DecoderBlock {
                weight: get(&format!("decoder.blocks.{i}.conv.weight"))?.to_device(device).map_err(|e| e.to_string())?,
                bias: get(&format!("decoder.blocks.{i}.conv.bias"))?.to_device(device).map_err(|e| e.to_string())?,
                norm_weight: get(&format!("decoder.blocks.{i}.norm.weight"))?.to_device(device).map_err(|e| e.to_string())?,
                norm_bias: get(&format!("decoder.blocks.{i}.norm.bias"))?.to_device(device).map_err(|e| e.to_string())?,
                stride,
                padding,
            });
        }
        let proj_weight = get("decoder.proj.weight")?.to_device(device).map_err(|e| e.to_string())?;
        let proj_bias = get("decoder.proj.bias")?.to_device(device).map_err(|e| e.to_string())?;

        Ok(Self { codebooks, codebook_proj, blocks, proj_weight, proj_bias, cfg, device: device.clone() })
    }

    /// codes: (N_q, T) or (B, N_q, T) i64 token indices -> mono f32 PCM samples.
    pub fn decode(&self, codes: &Tensor) -> Result<Vec<f32>, String> {
        let codes = if codes.dims().len() == 2 { codes.unsqueeze(0).map_err(|e| e.to_string())? } else { codes.clone() };
        let (b, n_q, t) = codes.dims3().map_err(|e| e.to_string())?;

        // RVQ decode: sum codebook embeddings across the N_q stack -> (B,T,D)
        let mut z_q = Tensor::zeros((b, t, self.cfg.codebook_dim), DType::F32, &self.device).map_err(|e| e.to_string())?;
        for q in 0..n_q {
            let idx = codes.narrow(1, q, 1).map_err(|e| e.to_string())?.reshape((b, t)).map_err(|e| e.to_string())?;
            let idx_flat = idx.reshape(b * t).map_err(|e| e.to_string())?;
            let looked_up = self.codebooks[q].index_select(&idx_flat, 0).map_err(|e| e.to_string())?; // (b*t, D)
            let looked_up = looked_up.reshape((b, t, self.cfg.codebook_dim)).map_err(|e| e.to_string())?;
            z_q = (z_q + looked_up).map_err(|e| e.to_string())?;
        }
        // (B,T,D) -> (B,D,T)
        let mut x = z_q.permute((0, 2, 1)).map_err(|e| e.to_string())?.contiguous().map_err(|e| e.to_string())?;

        if let Some((w, bi)) = &self.codebook_proj {
            x = conv1d_1x1(&x, w, bi).map_err(|e| e.to_string())?;
        }

        for blk in &self.blocks {
            x = x
                .conv_transpose1d(&blk.weight, blk.padding, 0, blk.stride, 1, 1)
                .map_err(|e| e.to_string())?;
            let out_ch = blk.bias.dims1().map_err(|e| e.to_string())?;
            x = x.broadcast_add(&blk.bias.reshape((1, out_ch, 1)).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            x = group_norm_1group(&x, &blk.norm_weight, &blk.norm_bias, 1e-5).map_err(|e| e.to_string())?;
            x = elu(&x).map_err(|e| e.to_string())?;
        }

        // Final Conv1d(kernel=7, padding=3) + tanh * 0.95
        let out_ch = self.proj_bias.dims1().map_err(|e| e.to_string())?;
        x = x.conv1d(&self.proj_weight, 3, 1, 1, 1).map_err(|e| e.to_string())?;
        x = x.broadcast_add(&self.proj_bias.reshape((1, out_ch, 1)).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        x = (x.tanh().map_err(|e| e.to_string())? * 0.95).map_err(|e| e.to_string())?;

        // (B=1, 1, T) -> Vec<f32>
        let flat = x.flatten_all().map_err(|e| e.to_string())?;
        flat.to_vec1::<f32>().map_err(|e| e.to_string())
    }

    pub fn sample_rate(&self) -> usize {
        self.cfg.sample_rate
    }
}

fn conv1d_1x1(x: &Tensor, weight: &Tensor, bias: &Tensor) -> candle_core::Result<Tensor> {
    let out = x.conv1d(weight, 0, 1, 1, 1)?;
    let out_ch = bias.dims1()?;
    out.broadcast_add(&bias.reshape((1, out_ch, 1))?)
}

#[derive(Deserialize, Default, Clone)]
pub struct VoicePackMetadata {
    #[serde(default)]
    pub name: String,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

pub struct VoicePack {
    pub metadata: VoicePackMetadata,
    pub decoder: Option<CodecDecoder>,
}

impl VoicePack {
    pub fn load(path: &Path, device: &Device) -> Result<Self, String> {
        let tmp_dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let gz = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(gz);
        archive.unpack(tmp_dir.path()).map_err(|e| e.to_string())?;

        let meta_path = tmp_dir.path().join("metadata.json");
        let metadata = if meta_path.exists() {
            let text = std::fs::read_to_string(&meta_path).map_err(|e| e.to_string())?;
            serde_json::from_str(&text).unwrap_or_default()
        } else {
            VoicePackMetadata::default()
        };

        let decoder_path = tmp_dir.path().join("decoder.pt");
        let decoder = if decoder_path.exists() { Some(CodecDecoder::load(&decoder_path, device)?) } else { None };

        Ok(Self { metadata, decoder })
    }

    /// Placeholder text->token synthesis, copied from VoicePack.synthesize()
    /// in the Python original -- explicitly not real TTS (see module docs).
    pub fn synthesize(&self, text: &str, device: &Device) -> Result<(Vec<f32>, usize), String> {
        let Some(decoder) = &self.decoder else {
            return Err("No codec decoder loaded in voice pack".to_string());
        };
        let text_chars: Vec<i64> = text.chars().take(200).map(|c| (c as i64).min(1023)).collect();
        let n_frames = (text_chars.len() * 3).max(10);
        let n_q = 4;
        let mut codes = vec![0i64; n_q * n_frames];
        for (i, &ch) in text_chars.iter().enumerate().take(n_frames) {
            for q in 0..n_q {
                codes[q * n_frames + i] = ch % 1024;
            }
        }
        let codes_t = Tensor::from_vec(codes, (n_q, n_frames), device).map_err(|e| e.to_string())?;
        let audio = decoder.decode(&codes_t)?;
        Ok((audio, decoder.sample_rate()))
    }
}

pub struct VoiceEngine {
    pub voices: HashMap<String, VoicePack>,
    pub active_voice: Option<String>,
    pub sample_rate: usize,
    device: Device,
    packs_dir: PathBuf,
}

impl VoiceEngine {
    pub fn new(packs_dir: PathBuf) -> Self {
        Self { voices: HashMap::new(), active_voice: None, sample_rate: 24000, device: Device::Cpu, packs_dir }
    }

    pub fn list_available_packs(&self) -> Vec<serde_json::Value> {
        let Ok(entries) = std::fs::read_dir(&self.packs_dir) else { return Vec::new() };
        let mut packs = Vec::new();
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().and_then(|x| x.to_str()) != Some("voice") {
                continue;
            }
            let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            packs.push(serde_json::json!({ "name": name, "path": path.to_string_lossy(), "loaded": self.voices.contains_key(&name) }));
        }
        packs
    }

    pub fn load_pack(&mut self, path: &Path) -> Result<String, String> {
        if !path.exists() {
            return Err(format!("Voice pack not found: {}", path.display()));
        }
        let pack = VoicePack::load(path, &self.device)?;
        let name = if !pack.metadata.name.is_empty() {
            pack.metadata.name.clone()
        } else {
            path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()
        };
        self.sample_rate = pack.decoder.as_ref().map(|d| d.sample_rate()).unwrap_or(24000);
        self.voices.insert(name.clone(), pack);
        if self.active_voice.is_none() {
            self.active_voice = Some(name.clone());
        }
        Ok(name)
    }

    pub fn activate(&mut self, name: &str) -> Result<(), String> {
        if !self.voices.contains_key(name) {
            return Err(format!("Voice '{name}' not loaded"));
        }
        self.active_voice = Some(name.to_string());
        Ok(())
    }

    pub fn synthesize(&self, text: &str, voice: Option<&str>) -> Result<(Vec<f32>, usize), String> {
        let v = voice.map(String::from).or_else(|| self.active_voice.clone());
        let Some(v) = v else { return Err("No voice loaded. Load a voice pack first.".to_string()) };
        let pack = self.voices.get(&v).ok_or("No voice loaded. Load a voice pack first.")?;
        pack.synthesize(text, &self.device)
    }
}
