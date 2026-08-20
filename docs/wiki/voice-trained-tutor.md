# Voice-Trained Tutor: Training an Agent from a Real Teacher's Audio

Goal: an AI math tutor whose pedagogy (not just content) comes from a real human
teacher — trained on their **audio**, so prosody, pacing, and encouragement
(inflection) are learned, not synthesized. Research summary, pipeline, and
recording spec. No AI-generated "teaching style" — the teacher's own voice and
moves are the training signal.

## Why audio (not text)

Text fine-tuning captures *what* a teacher says but not *how*. Full-duplex
speech models trained on spoken dialogue learn paralinguistics directly (tone,
timing, overlap, encouragement) because there is no text bottleneck. OpenAI's
GPT-4o / Google's Gemini Live do this closed-source; open options exist.

## Open models that support full-audio fine-tuning

| Model | Why | Training |
|---|---|---|
| **Moshi** (Kyutai, MIT) | Speech-text foundation model; full-duplex (handles overlap/interruption); prosody learned end-to-end | `kyutai-labs/moshi-finetune`: stereo audio (teacher L / student R) → dataset → LoRA. 7B, one A100 80GB comfortable |
| **Qwen3-Omni** (30B-A3B) | Stronger general model, audio in/audio out | LoRA/QLoRA (reportedly fits 24GB), ms-swift / LLaMA-Factory; data = interleaved audio clips + text JSONL |
| **MiniCPM-o 2.6** | 8B, lighter omni model | Audio fine-tuning via LLaMA-Factory (`mllm_audio_demo` format) |

Closed (no fine-tuning access): GPT-4o, Gemini Live.

## The verbal training loop

```
1. RECORD      Real teacher + student, two mics, stereo (see spec below).
2. AUTO-DATA   moshi-finetune: VAD → align channels → Mimi audio tokens.
               Optional ASR transcripts = auxiliary text stream (stabilizes
               training, enables text prompts too).
3. TRAIN       LoRA SFT on Moshi (~7B). Start with 10–50 hrs of audio.
4. EVAL        Listening is the only honest eval: teacher rates generated
               sessions (blinded) vs base model.
5. ITERATE     Bad turns → find similar real audio → add → retrain.
```

## Single-mic fallback: speaker diarization

Diarization ("who spoke when") is standard in WhisperX (open, local, pyannote),
Otter, Rev, Fireflies, Descript, MS Word transcribe, newer Whisper.

Key rule: **the training signal is the audio slices, not the transcript.**
Cut the original audio at diarization boundaries and reassemble into
teacher-channel / student-channel stereo. Transcript is only the auxiliary text.

Caveats:
- Single channel loses overlaps (two people talking at once) — stereo avoids this.
- Diarization errors are training poison: filter by confidence, have the
  teacher spot-check boundaries before training.

## Recording spec (stereo path — makes diarization optional)

- **Mics**: two lavalier (or headset) mics — teacher on L channel, student on R.
  Avoid one room mic (bleed, and single channel).
- **Format**: WAV, 16-bit PCM, **44.1 kHz or 48 kHz** (Moshi's codec expects
  24 kHz internally but upsamples; record at the standard rate, convert in the
  pipeline). Raw WAV preferred — no MP3/Opus lossy on the training signal.
- **Channels**: dual mono, hard-panned. Record into a field recorder or phone
  app that writes true two-track (e.g., USB interface + any DAW/recorder).
- **Levels**: keep peaks around −6 to −3 dBFS; no clipping (clipping teaches
  the model to clip). ~30–45 min sessions are a good unit.
- **Privacy**: real students are minors — keep audio local; WhisperX/diarization
  and training all run on your own GPU so data never leaves the machine.

## Serving

Voice models do NOT run on the 1bit-MONSTER text engine — serving is a separate
stack (Moshi's own inference / vLLM for Qwen3-Omni). Training is GPU work
regardless; the engine is irrelevant until a text-only distillation is wanted.

## Ceilings

- Voice identity (sounding exactly like the teacher) is the weak spot; style
  transfer works with 10–50 hrs, exact voice cloning needs much more — pair
  with a cloned TTS (F5-TTS/CosyVoice) if the exact voice is required, at the
  cost of end-to-end inflection control.
- Imitation teaches surface behavior. Eval against real learning outcomes
  before trusting it with students.

## Related text-only work (context)

ConvoLearn (Stanford, MIT, `masharma/convolearn` on HF): 2,134 dialogues typed
by 323 credentialed teachers against a simulated student (Gemini-1.5-Pro),
labeled across 6 dialogic dimensions (cognitive engagement, formative
assessment, accountability, cultural responsiveness, metacognition, power
dynamics), with effectiveness/completeness ratings. QLoRA on Mistral-7B over
the 1,250-dialogue HQ subset ≈ Claude Sonnet 4.5 in blinded teacher ratings.
Format lesson: **progressive samples** — predict each teacher turn given all
prior context. Same filtering lesson: keystroke-rate check to catch pasted
AI text; dual-annotator quality screening.
