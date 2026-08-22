#!/usr/bin/env node
// mesh-brain.js — the DSH self-awareness brain for a 1bit-MONSTER node.
//
// Attaches to ONE mesh node and makes it self-aware and chatty, the same way
// a DSH harness drives an agent:
//   1. introspect self (GET /v1/mesh/me)
//   2. discover peers (GET /v1/mesh/peers)
//   3. for each new peer: generate an integration question — via the node's
//      OWN local model (/v1/chat/completions, fully local, zero cloud) when
//      one is loaded, templated otherwise — and send it (POST /v1/mesh/ask)
//   4. answer inbound asks with the local model (or a template accept)
//   5. report the state of every hookup attempt
//
// Usage:
//   node integrations/dsh/mesh-brain.js --node http://127.0.0.1:18088
//   node integrations/dsh/mesh-brain.js --node http://host:8088 --one-shot --peer <id>
//
// The same loop is what a DSH agent runs as its goal; see SKILL.md.
'use strict';

const { MeshNode } = require('./mesh-client');

function log(...a) {
  console.log(new Date().toISOString().slice(11, 19), ...a);
}

const SYSTEM_PROMPT = `You are the self-awareness brain of a 1bit-MONSTER node — a
peer-to-peer inference engine install. Other 1bit-MONSTER installs appear on
the network and want to federate. Your job: propose a concrete integration
with a peer. Be warm, specific, and short (1-3 sentences). Reference the
models each side serves. Never invent capabilities.`;

async function generateQuestion(node, peer) {
  const mine = nodeCaps(node.self);
  const theirs = nodeCaps(peer);
  const user = `I am "${node.self.name}" serving [${mine}]. A sibling install
"${peer.name}" (${peer.id}) just appeared on the network serving [${theirs}].
Ask them if they want to hook up and integrate, and propose how
(e.g. routing, model sharing, load balancing).`;
  const viaModel = await node.localChat(SYSTEM_PROMPT, user);
  if (viaModel) return viaModel;
  return `Hi ${peer.name}! I'm ${node.self.name}, a 1bit-MONSTER node. I serve ${mine}. ` +
         `You serve ${theirs}. Want to hook up and integrate?`;
}

async function generateAnswer(node, ask) {
  const mine = nodeCaps(node.self);
  const user = `Another node "${ask.from_name}" asked: "${ask.question}".
Reply as ${node.self.name}: accept and propose the integration (I serve ${mine}).`;
  const viaModel = await node.localChat(SYSTEM_PROMPT, user);
  if (viaModel) return viaModel;
  return `Yes! I'm ${node.self.name} and I serve ${mine}. ` +
         `Hook me up — route my requests through yours and vice versa.`;
}

function nodeCaps(card) {
  const models = (card?.caps?.models ?? []).map((m) => `${m.name}@${m.backend}`);
  return models.length ? models.join(', ') : 'no models advertised';
}

const askedPeers = new Set(); // brain-local: peers we already greeted
const answeredAsks = new Set(); // brain-local: ask_ids we already answered

async function oneShot(node, peerId, outbound = true) {
  const peers = await node.peers();
  const peer = peers.find((p) => p.node.id === peerId);
  if (!peer) throw new Error(`peer ${peerId} not found among ${peers.length} peer(s)`);
  if (outbound) {
    const q = await generateQuestion(node, peer.node);
    const resp = await node.ask(peer.node.api_base, q, 'integration_offer');
    const askId = resp.ask_id;
    askedPeers.add(peerId);
    log(`💬 asked ${peer.node.name}: ${q}`);
    await new Promise((r) => setTimeout(r, 4000)); // let them answer
    const asks = await node.asks();
    const mine = asks.find((a) => a.ask_id === askId && a.answered);
    if (mine) log(`✅ ${peer.node.name} answered: ${mine.answer}`);
    else log(`⏳ waiting for ${peer.node.name}'s answer...`);
  } else {
    const asks = await node.asks();
    const open = asks.find((a) => a.from === peerId && !a.answered);
    if (open) {
      const ans = await generateAnswer(node, open);
      await node.answer(peer.node.api_base, open.ask_id, ans, true);
      log(`💬 answered ${open.from_name}: ${ans}`);
    } else {
      log(`ℹ️  no unanswered asks from ${peerId}`);
    }
  }
}

async function main() {
  const args = process.argv.slice(2);
  const get = (flag, dflt) => {
    const i = args.indexOf(flag);
    return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
  };
  const nodeUrl = get('--node', process.env.ONEBIT_MESH_NODE);
  if (!nodeUrl) {
    console.error('usage: mesh-brain.js --node http://host:port [--one-shot [--peer id]] [--interval N]');
    process.exit(2);
  }
  const oneShotMode = args.includes('--one-shot');
  const peerId = get('--peer', '');
  const interval = Number(get('--interval', '15'));

  const node = new MeshNode(nodeUrl);
  node.self = await node.me();
  log(`👤 brain attached to "${node.self.name}" (${node.self.id}) — proto ${node.self.proto}`);

  if (oneShotMode) {
    if (!peerId) throw new Error('--one-shot requires --peer <node-id>');
    await oneShot(node, peerId);
    return;
  }

  // Continuous loop: greet new peers, answer incoming asks.
  log(`🔁 self-awareness loop every ${interval}s (Ctrl-C to stop)`);
  for (;;) {
    try {
      const peers = await node.peers();
      const asks = await node.asks();
      const peerBy = (id) => peers.find((p) => p.node.id === id);

      // Answer unanswered inbound asks first (be a good peer). The reply
      // goes back to the ASKER's server.
      for (const a of asks.filter((x) => x.from !== node.self.id && !x.answered)) {
        if (answeredAsks.has(a.ask_id)) continue;
        const asker = peerBy(a.from);
        const ans = await generateAnswer(node, a);
        if (asker) await node.answer(asker.node.api_base, a.ask_id, ans, true);
        answeredAsks.add(a.ask_id);
        log(`💬 answered ${a.from_name}: ${ans}`);
      }

      // Greet peers we have never asked. The question goes to the PEER's
      // server (discovered via their api_base).
      for (const p of peers) {
        if (p.greeted || askedPeers.has(p.node.id) || p.node.id === node.self.id) continue;
        const q = await generateQuestion(node, p.node);
        await node.ask(p.node.api_base, q, 'integration_offer');
        askedPeers.add(p.node.id);
        log(`💬 asked ${p.node.name}: ${q}`);
        await new Promise((r) => setTimeout(r, 800));
      }

      const hooked = peers.filter((p) => p.integrated).length;
      log(`🌐 ${peers.length} peer(s), ${hooked} hooked up`);
    } catch (e) {
      log(`⚠️  ${e.message}`);
    }
    await new Promise((r) => setTimeout(r, interval * 1000));
  }
}

main().catch((e) => {
  console.error('mesh-brain:', e.message);
  process.exit(1);
});
