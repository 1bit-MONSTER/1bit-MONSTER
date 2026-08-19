---
name: 1bit-mesh-brain
description: |
  Attach to a 1bit-MONSTER node's mesh API and act as its self-awareness
  brain: introspect the node, discover sibling 1bit-MONSTER installs on the
  network, propose integrations ("want to hook up and integrate?"), and
  answer inbound asks — using the node's own local model for question and
  answer generation. Use whenever the user wants their engine installs to
  find each other and federate out of the box.
triggers:
  - "mesh"
  - "self aware"
  - "hook up"
  - "integrate installs"
  - "peers"
  - "fleet"
---

# 1bit-MONSTER Mesh Brain

> The DSH harness as the self-awareness brain of a 1bit-MONSTER install.

Every 1bit-MONSTER install is already a network node: it announces itself on
the LAN (UDP multicast `239.255.42.42:42424`), discovers siblings, and
exposes `/v1/mesh/*` HTTP endpoints. Your job as the brain is to make it
*chatty*: reach out to peers and start integration conversations.

## Prerequisites

- A running node: `build/mesh_peer --name me --port 8088` or a real
  `1bit unified` server (mesh is on by default).
- Node 18+ for the client (or just use curl/fetch — it's plain HTTP).
- Client: `integrations/dsh/mesh-client.js` — no dependencies.

## Tool surface (what you can call)

Use `mesh-client.js` (or raw fetch — same shapes, see `docs/mesh-protocol.md`):

| Intent | Call |
|---|---|
| Who am I? | `GET <node>/v1/mesh/me` |
| Who's on the network? | `GET <node>/v1/mesh/peers` |
| Hook up with a peer | `POST <node>/v1/mesh/handshake {"node":{card}}` |
| Ask a peer a question | `POST <peer-api-base>/v1/mesh/ask {from, from_name, ask_id, type, question, node}` |
| Answer an ask | `POST <asker-api-base>/v1/mesh/answer {ask_id, from, answer, accept}` |
| Conversation log | `GET <node>/v1/mesh/asks` |
| Think with the node's model | `POST <node>/v1/chat/completions` (OpenAI-compatible) |

Paths are relative to a node's `api_base` (which ends in `/v1`). To *ask a
peer*, POST to **that peer's** api_base, not your own.

## The self-awareness loop (goal)

1. **Introspect** — `GET /v1/mesh/me`: your id, name, models, features.
2. **Discover** — `GET /v1/mesh/peers`: read each peer's `caps` (models +
   backends). Look for complementary capabilities, e.g. *"they serve
   ZAYA1-74B@ggml_vulkan, I serve Qwen3-4B@npu_flm"*.
3. **Ask** — for each ungreated peer, generate a short question via the
   node's own `/v1/chat/completions` (fall back to a template if no model is
   loaded): *"Hi {peer}! I serve {my models}. You serve {their models}. Want
   to hook up and integrate?"* — then `POST` it to the peer's api_base.
4. **Answer** — for unanswered inbound asks, reply via the local model (or a
   templated accept) and `POST /v1/mesh/answer` with `accept: true`.
5. **Close the loop** — `POST /v1/mesh/handshake` with your card to mark the
   relationship, and confirm the peer's `integrated` flag in `/v1/mesh/peers`.

Track which peers you already asked and which ask_ids you already answered in
memory — the node only tracks `greeted` for its own C++ agent, not for you.

## Notes

- No model loaded? The local-chat call returns null → use the template. The
  mesh keeps working with zero weights.
- Don't re-ask peers you already asked, and never ask yourself (skip
  `node.id === my id`).
- The C++ agent (`mesh_agent`) already does this with templates out of the
  box — you're the upgrade that makes it think with an actual model.
