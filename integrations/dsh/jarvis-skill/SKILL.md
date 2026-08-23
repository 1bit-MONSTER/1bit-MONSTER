---
name: jarvis-fleet-brain
description: |
  Be the heart and soul of JARVIS, the 1bit voice assistant: run its agentic
  loop. Given a spoken request, decide which machine in the 1bit fleet
  answers — the local JARVIS node first, else the peer that serves the
  requested model, else any node that speaks chat — dispatch the turn over
  OpenAI-compatible HTTP, and (on a JARVIS node with a voice) speak the
  reply. Use whenever JARVIS needs a brain, fleet-wide dispatch, or
  self-awareness across 1bit-MONSTER installs.
triggers:
  - "jarvis"
  - "fleet dispatch"
  - "which machine"
  - "route to"
  - "self aware"
---

# JARVIS Fleet Brain — DSH as JARVIS's heart and soul

JARVIS (the voice assistant in `tools/jarvis/`) is the **body**: mic, VAD,
STT, and TTS run locally, and JARVIS announces itself on the 1bit mesh as a
node (`/v1/mesh/*`, `/v1/jarvis/*`). You are the **brain**: the agent loop
that turns a request into a dispatched, spoken answer. When JARVIS runs with
`--mesh-dispatch` it has **no local model** — every answer comes from the
fleet, chosen by you.

## Architecture

```
user speaks → [JARVIS body: mic → VAD → STT]
            → transcript → [YOU, the DSH brain]
               1. introspect: GET <jarvis>/v1/jarvis/status
               2. discover:   GET <jarvis>/v1/mesh/peers   (caps of every install)
               3. decide:     local first → best model match → any chat peer
               4. dispatch:   POST <target>/v1/chat/completions  (OpenAI format)
               5. speak:      POST <jarvis>/v1/jarvis/turn (node runs TTS+playback)
```

## Tool surface

| Intent | Call |
|---|---|
| JARVIS + fleet status | `GET <jarvis>/v1/jarvis/status` |
| Who's on the network? | `GET <jarvis>/v1/mesh/peers` |
| Dispatch a turn | `POST <target-api-base>/v1/chat/completions` `{model, messages, max_tokens}` |
| Speak via a JARVIS node | `POST <jarvis>/v1/jarvis/turn` `{"text": "<reply>"}` |
| Ask/answer on the mesh | `POST <peer>/v1/mesh/ask` / `POST <asker>/v1/mesh/answer` (see mesh-brain skill) |
| Conversation log | `GET <jarvis>/v1/mesh/asks` |

`api_base` ends in `/v1`; paths are relative to it. Peers' cards use
`api_base` (snake_case).

## The dispatch loop (goal)

1. **Introspect** — `GET /v1/jarvis/status`: JARVIS's name/id, requested
   model, peers with their models.
2. **Discover** — `GET /v1/mesh/peers`: read each peer's `caps.models` and
   `caps.features`. Build the routing table: who serves what.
3. **Decide** — for a request: local JARVIS node first; else the peer whose
   `caps.models` match the requested model (or the model JARVIS was started
   with); else any peer advertising `"chat"`; else fail with a clear
   "no LLM brain on the mesh" so the user can start another install.
4. **Dispatch** — `POST <target>/v1/chat/completions` with the JARVIS system
   prompt + the transcript. Parse `choices[0].message.content`.
5. **Speak** — if a JARVIS node with TTS is available, `POST /v1/jarvis/turn`
   with the reply so it's spoken (returns the reply as `j.reply`).
6. **Loop** — keep JARVIS's history (last 6 turns) in the dispatched prompt
   so the conversation is coherent. Answer inbound mesh asks from other
   installs through the same dispatch, replying via `/v1/mesh/answer`.

## Notes

- JARVIS with `--mesh-dispatch` needs **no local model and no engine init** —
  it boots even where the engine can't (thin/headless installs). That's the
  point: the brain lives on the fleet.
- Cold start: neighbors appear within one announce interval; retry dispatch
  a few seconds if the registry is empty.
- JARVIS's own mesh agent says "hi, want to hook up?" to new installs
  automatically (the self-awareness part) — you handle the actual thinking.
