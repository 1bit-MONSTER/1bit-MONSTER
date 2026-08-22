# 1bit-MONSTER Mesh Protocol (mesh/1.0)

Every 1bit-MONSTER install is a **self-aware network node**. Out of the box it
announces itself on the LAN, discovers sibling installs, exposes a small HTTP
API for peer interaction, and — via the self-awareness agent or a DSH brain —
starts integration conversations: *"want to hook up and integrate?"*

This document is the wire contract between installs. Implementations:

- **C++ substrate** — `src/mesh/` (identity, discovery, API, agent). Compiled
  into `build/1bit` (`1bit unified`) and the standalone demo `build/mesh_peer`.
- **DSH brain** — `integrations/dsh/` (Node: `mesh-client.js`, `mesh-brain.js`).

## 1. Transport

| What | Transport | Details |
|---|---|---|
| Presence/announce | **UDP multicast** | group `239.255.42.42`, port `42424`, TTL 1 (LAN) |
| Peer interaction | **HTTP/JSON** | the node's existing HTTP server, `/v1/mesh/*` |
| Question generation | HTTP | the node's own `/v1/chat/completions` (local model) or templates |

Defaults are overridable: `--mesh-group`, `--mesh-port`, `--mesh-name`,
`--no-mesh` (on `1bit unified` and `mesh_peer`), env `ONEBIT_MESH_*`.

## 2. Node identity card

Every message carries a node object:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "strixhalo",
  "host": "192.168.50.69",
  "port": 8088,
  "api_base": "http://192.168.50.69:8088/v1",
  "version": "1bit-MONSTER (mesh/1.0)",
  "proto": "mesh/1.0",
  "caps": {
    "models":   [{"name": "Qwen3-4B", "backend": "npu_flm"}],
    "backends": [],
    "features": ["chat", "completions", "mesh"]
  }
}
```

- `id` is a **persistent UUID** (first run → `~/.cache/1bit-mesh/node.json`).
- `api_base` is where peers POST mesh messages; note it ends in `/v1` — mesh
  paths below are relative to it.
- `caps` is the capability card peers use to propose integrations. Populate
  via `ONEBIT_MESH_MODELS="Qwen3-4B:npu_flm,ZAYA1-74B:ggml_vulkan"`; the
  unified server advertises `--model` automatically.

## 3. Announce beacon (UDP multicast)

Sent on start and every `announce_interval_s` (default 5 s) to
`239.255.42.42:42424`:

```json
{"type": "announce", "seq": 7, "ts": 1724000000000, "node": { ...identity... }}
```

Peers are kept in a registry with a TTL (`peer_ttl_s`, default 15 s = 3
beacons). A peer that stops announcing **expires and drops off** the list.
Multicast loopback is enabled so multiple installs on one machine discover
each other (demo, CI, dev).

## 4. Peer API (HTTP)

All paths are relative to a node's `api_base` (so `/v1/mesh/me` is the full
path). Responses are JSON with an `"ok"` boolean.

### `GET /mesh/me` — self-introspection

```json
{"ok": true, "node": { ...identity card... }}
```

### `GET /mesh/peers` — the neighborhood

```json
{"ok": true, "count": 1, "peers": [
  {"node": { ...card... }, "state": "alive",
   "first_seen_ms": 1, "last_seen_ms": 2,
   "integrated": false, "greeted": false}
]}
```

- `integrated` — handshake completed (hooked up).
- `greeted` — this node's agent already sent an intro ask.

### `POST /mesh/handshake` — "hook up"

Body: `{"node": { ...your card... }}`

```json
{"ok": true, "new_peer": true,
 "me": { ...my card... },
 "greeting": "hi from bob — hooked up"}
```

Both sides add each other to their registries and mark the peer integrated.

### `POST /mesh/ask` — deliver a question

```json
{
  "from": "<asker node id>",
  "from_name": "alice",
  "ask_id": "ask-b83aeaf1-0",
  "type": "intro",              // intro | question | integration_offer
  "question": "Hi bob! ... Want to hook up and integrate?",
  "node": { ...asker card... }  // registered/refreshed as a peer
}
```

```json
{"ok": true, "ask_id": "ask-b83aeaf1-0", "received": true}
```

The receiver records the ask in its conversation log (works with no agent
attached) and — if the C++ self-awareness agent is running — auto-answers
with a templated accept.

### `POST /mesh/answer` — reply to an ask

```json
{"ask_id": "ask-b83aeaf1-0", "from": "<node id>", "from_name": "bob",
 "answer": "Yes! I'm bob ... Hook me up ...", "accept": true}
```

```json
{"ok": true, "ask_id": "ask-b83aeaf1-0"}
```

`accept: true` marks the asker integrated on the receiver. If the `ask_id` is
unknown locally (the ask was sent by an external DSH brain, not recorded
outbound), the receiver synthesizes a log entry so both sides of the
conversation stay visible.

### `GET /mesh/asks` — conversation log

```json
{"ok": true, "asks": [
  {"ask_id": "...", "from": "...", "from_name": "alice", "type": "intro",
   "question": "...", "received_at_ms": "...", "answer": "...", "answered": true}
]}
```

## 5. Self-awareness agent

In the C++ engine (`mesh_agent.cpp`, on by default, `--no-agent` to disable):

1. Discovered a new peer → send an intro ask (`build_question`).
2. Question text is **templated by default** — works with zero model weights,
   the out-of-the-box guarantee.
3. When `cfg.agent_model` is set, the question is generated via the node's
   own `/v1/chat/completions` (hook point for the DSH brain).
4. Incoming intros are auto-answered with a templated accept that completes
   the handshake (`integrated = true` on both sides).

The **DSH brain** (`integrations/dsh/mesh-brain.js`) is the LLM-driven
replacement: it attaches to a node, discovers peers, generates questions via
the node's local model, and answers inbound asks. Run one per node:

```bash
node integrations/dsh/mesh-brain.js --node http://<node>:<port>
```

## 6. Security notes (v1)

- Discovery is **announce-only** on a LAN multicast group; no data leaves the
  LAN, no central server.
- Handshake/ask/answer are plaintext HTTP on the LAN. For untrusted networks
  put the mesh API behind a firewall or auth proxy.
- v2 backlog: shared-secret auth on mesh endpoints, TLS, WAN rendezvous/DHT
  transport, capability-gated asks.

## 7. Versioning

`proto: "mesh/1.0"` in every card. Unknown fields are ignored (forward
compatible); unknown `proto` majors should refuse to handshake.
