# DSH × 1bit-MONSTER — the self-aware brain

[DSH](https://github.com/deepseek-ai/deepseek-harness) (DeepSeek Harness) is
the agent harness. 1bit-MONSTER is a pure-C++ inference engine. This is the
bridge: the engine is the **body** (network presence, peer discovery,
`/v1/mesh/*` API — works with zero extra deps), DSH is the **brain** (LLM-
driven self-awareness: discovering sibling installs and starting
integration conversations).

```
┌─ engine (C++, out of the box) ──────────────┐
│ node_identity · peer_discovery (multicast)   │
│ peer_api (/v1/mesh/*) · mesh_agent (templ.)  │
└───────────────┬──────────────────────────────┘
                │ HTTP
┌───────────────▼──────────────────────────────┐
│ DSH brain (Node, integrations/dsh)           │
│ mesh-client.js — API client (no deps)        │
│ mesh-brain.js — self-awareness loop          │
│ SKILL.md      — the loop as a DSH skill      │
└──────────────────────────────────────────────┘
```

## Quick start

```bash
# 1. run two nodes (no model weights needed)
cmake --build build --target mesh_peer -j
./build/mesh_peer --name alice --port 18088 --state-dir /tmp/a
./build/mesh_peer --name bob   --port 18089 --state-dir /tmp/b

# 2. they already found each other — verify
curl http://127.0.0.1:18088/v1/mesh/peers

# 3. attach a DSH brain to bob (continuous loop), then ask from alice
node integrations/dsh/mesh-brain.js --node http://127.0.0.1:18089 --interval 3 &
BOB_ID=$(curl -s http://127.0.0.1:18089/v1/mesh/me | jq -r .node.id)
node integrations/dsh/mesh-brain.js --node http://127.0.0.1:18088 \
     --one-shot --peer "$BOB_ID"
```

With a real `1bit unified` server the mesh is **on by default** (announces,
discovers, greets) — `--no-mesh` opts out. Advertise what you serve via
`--model NAME` or `ONEBIT_MESH_MODELS="Qwen3-4B:npu_flm,..."`.

## JARVIS — DSH as the voice assistant's heart and soul

`jarvis-brain.js` + the `jarvis-fleet-brain` skill make DSH the brain of the
JARVIS voice pipeline. JARVIS (the body: mic → VAD → STT → TTS) runs with
`--mesh-dispatch` and **no local model**; every LLM turn is dispatched to
the fleet node that serves the requested model:

```bash
# fleet node serving a model (real server, or a stub for testing):
./build/mesh_peer --name alice --port 18088 --stub-chat --models "Qwen3-4B:stub"

# thin JARVIS: local voice, brain on the mesh (no engine init needed):
./build/1bit jarvis --mesh-dispatch --text --model Qwen3-4B --port 18081

# the DSH brain routes capability-aware:
node integrations/dsh/jarvis-brain.js --node http://127.0.0.1:18081 --fleet
node integrations/dsh/jarvis-brain.js --node http://127.0.0.1:18081 \
     --say "what can you do?" --model Qwen3-4B
```

JARVIS itself is a fleet citizen: it announces on the mesh, exposes
`/v1/jarvis/turn` (text in → dispatched reply out, spoken if a voice is
loaded), and its mesh agent greets new installs automatically.

## Files

| File | Purpose |
|---|---|
| `mesh-client.js` | MeshNode HTTP client — `me()`, `peers()`, `handshake()`, `ask()`, `answer()`, `asks()`, `localChat()` |
| `mesh-brain.js` | The self-awareness loop: introspect → discover → ask → answer; `--one-shot --peer <id>` for a single greeting |
| `jarvis-brain.js` | The JARVIS fleet brain: `--fleet` status, `--say "text" --model X` capability dispatch, `--listen` answers mesh asks |
| `SKILL.md` | The mesh self-awareness loop as a DSH skill |
| `jarvis-skill/SKILL.md` | The JARVIS fleet dispatch loop as a DSH skill |

## Local-model thinking

`mesh-brain.js` generates questions/answers through the node's **own**
`/v1/chat/completions` — fully local, zero cloud. When the node serves no
model it falls back to the template, so the mesh is never dead in the water.

## Wire contract

`docs/mesh-protocol.md` — announce beacon, identity card, and the
ask/answer/handshake schemas.
