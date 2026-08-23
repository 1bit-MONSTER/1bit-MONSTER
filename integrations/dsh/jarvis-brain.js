#!/usr/bin/env node
// jarvis-brain.js — DSH as JARVIS's heart and soul: fleet dispatch.
//
// JARVIS (the 1bit voice assistant) is the body: mic/STT/TTS local, and it
// announces itself on the mesh as a node. This brain is DSH: given a
// request, it decides WHICH node in the fleet should answer — the local
// one first, then the best peer by capability match (the machine that
// serves the requested model) — and dispatches the turn over plain
// OpenAI-compatible HTTP. When the target is a JARVIS node with a voice,
// the reply is spoken.
//
// Modes:
//   --say "text" [--model NAME]     one-shot: dispatch one request, print reply
//   --listen                         continuous: watch mesh asks, answer via fleet
//   --fleet                          print the fleet status table
//   --speak                          with --say/--listen: speak via a JARVIS node
//
// Usage:
//   node integrations/dsh/jarvis-brain.js --node http://127.0.0.1:8081 \
//        --say "what's the weather?" --model Qwen3-4B
'use strict';

const { MeshNode } = require('./mesh-client');

function log(...a) { console.log(new Date().toISOString().slice(11, 19), ...a); }

const JARVIS_SYSTEM =
  'You are JARVIS, the voice assistant of a 1bit-MONSTER fleet. ' +
  'Answer concisely, in 1-3 sentences, as speech.';

// ── Fleet dispatch: pick the best node and POST a chat completion ──────
async function dispatch(node, text, model) {
  const candidates = await rankCandidates(node, model);
  for (const c of candidates) {
    const r = await chatOn(c.apiBase, model, JARVIS_SYSTEM, text);
    if (r) return { ...r, node: c };
  }
  throw new Error(candidates.length
    ? `all ${candidates.length} candidate(s) failed`
    : 'no peers on the mesh');
}

// Local node first, then peers advertising the model, then any chat peer.
async function rankCandidates(node, model) {
  const peers = (await node.peers()).map((p) => p.node);
  // Identity cards use api_base (snake_case); normalize to apiBase.
  const card = (n) => ({ apiBase: n.api_base || n.apiBase, name: n.name || '?', id: n.id || '' });
  const local = { apiBase: node.baseUrl, name: (await node.me()).name, local: true };
  const offer = (n) => n.caps?.models?.some((m) => !model || m.name === model || m.name.startsWith(model));
  const chat = (n) => n.caps?.features?.includes('chat');
  const best = peers.filter(offer).map(card);
  const rest = peers.filter((n) => !offer(n) && chat(n)).map(card);
  return [local, ...best, ...rest];
}

async function chatOn(apiBase, model, system, user) {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 30000);
    const res = await fetch(apiBase + '/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: model || 'auto',
        messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
        max_tokens: 256,
        temperature: 0.6,
        stream: false,
      }),
      signal: ctl.signal,
    });
    clearTimeout(t);
    if (!res.ok) return null;
    const j = await res.json();
    const content = j?.choices?.[0]?.message?.content?.trim();
    return content ? { reply: content, model: j.model || model } : null;
  } catch {
    return null;
  }
}

// Speak via a JARVIS node's /v1/jarvis/turn (node runs TTS + playback).
async function speakVia(node, jarvisNodeApiBase, text) {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 60000);
    const res = await fetch(jarvisNodeApiBase + '/v1/jarvis/turn', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: ctl.signal,
    });
    clearTimeout(t);
    if (!res.ok) return null;
    const j = await res.json();
    return j.ok ? j.reply : null;
  } catch {
    return null;
  }
}

// ── Modes ───────────────────────────────────────────────────────────────
async function fleetTable(node) {
  const me = await node.me();
  const peers = await node.peers();
  log(`🌐 fleet — I am "${me.name}" (${me.id.slice(0, 8)})`);
  for (const p of peers) {
    const models = (p.node.caps?.models ?? []).map((m) => `${m.name}@${m.backend}`).join(', ') || '—';
    log(`   ${p.node.name.padEnd(12)} ${(p.node.host || '').padEnd(14)} ${models.padEnd(30)} integrated=${p.integrated}`);
  }
  if (!peers.length) log('   (no sibling installs discovered yet — wait for beacons)');
}

async function sayOnce(node, text, model, speak) {
  const r = await dispatch(node, text, model);
  log(`💬 [${r.node.local ? 'local' : r.node.name}/${r.model}] ${r.reply}`);
  if (speak) {
    const spoken = await speakVia(node, node.baseUrl, r.reply);
    if (spoken) log(`🔊 spoke on ${node.baseUrl}`);
  }
}

async function listenLoop(node, model, speak) {
  log(`👂 listening on the mesh — answering asks via the fleet (Ctrl-C to stop)`);
  const answered = new Set();
  for (;;) {
    try {
      const asks = await node.asks();
      for (const a of asks.filter((x) => !x.answered && !answered.has(x.ask_id))) {
        log(`📨 ${a.from_name} asks: ${a.question.slice(0, 120)}`);
        try {
          const r = await dispatch(node, a.question, model);
          log(`💬 → ${a.from_name}: ${r.reply.slice(0, 200)}`);
          // Reply via the asker's answer endpoint (accept = integrated).
          const peers = await node.peers();
          const asker = peers.find((p) => p.node.id === a.from);
          if (asker) {
            const me = await node.me();
            const res = await fetch(asker.node.api_base + '/mesh/answer', {
              method: 'POST',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify({ ask_id: a.ask_id, from: me.id, from_name: me.name, answer: r.reply, accept: true }),
            });
            if (!res.ok) log(`⚠️  answer delivery to ${a.from_name} failed`);
          }
          answered.add(a.ask_id);
        } catch (e) {
          log(`⚠️  ${e.message}`);
        }
      }
    } catch (e) {
      log(`⚠️  ${e.message}`);
    }
    await new Promise((r) => setTimeout(r, 5000));
  }
}

async function main() {
  const args = process.argv.slice(2);
  const get = (f, d) => { const i = args.indexOf(f); return i >= 0 && args[i + 1] ? args[i + 1] : d; };
  const nodeUrl = get('--node', process.env.ONEBIT_MESH_NODE);
  if (!nodeUrl) {
    console.error('usage: jarvis-brain.js --node http://jarvis-node:port [--say "text"] [--model NAME] [--speak] [--fleet] [--listen]');
    process.exit(2);
  }
  const model = get('--model', '');
  const speak = args.includes('--speak');
  const node = new MeshNode(nodeUrl);

  if (args.includes('--fleet')) { await fleetTable(node); return; }
  if (args.includes('--listen')) { await listenLoop(node, model, speak); return; }
  const text = get('--say', '');
  if (!text) { console.error('nothing to do: pass --say "text", --listen, or --fleet'); process.exit(2); }
  await sayOnce(node, text, model, speak);
}

main().catch((e) => { console.error('jarvis-brain:', e.message); process.exit(1); });
