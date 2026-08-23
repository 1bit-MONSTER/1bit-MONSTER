// SPDX-License-Identifier: MIT
//
// Adds the official 1bit.MONSTER #support channel (idempotent) and seeds a
// short welcome message. Reuses fluxer-mcp's own .env config loader so the
// token never has to be passed on the command line or printed.
//
// Usage:
//   node scripts/add-support-channel.mjs
//   node scripts/add-support-channel.mjs --fresh   # delete + recreate instead of reuse

import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// ── config (same .env precedence as dist/config.js) ─────────────────────────
const ENV = { ...process.env };
if (!ENV.FLUXER_TOKEN) {
  for (const file of ['.env', resolve(process.cwd(), '.env'), resolve(import.meta.dirname, '..', '.env')]) {
    if (!existsSync(file)) continue;
    for (const line of readFileSync(file, 'utf8').split('\n')) {
      const t = line.trim();
      if (!t || t.startsWith('#')) continue;
      const eq = t.indexOf('=');
      if (eq === -1) continue;
      const k = t.slice(0, eq).trim();
      const v = t.slice(eq + 1).trim().replace(/^["']|["']$/g, '');
      if (k && !(k in ENV)) ENV[k] = v;
    }
  }
}

const TOKEN = ENV.FLUXER_TOKEN;
const TOKEN_TYPE = ENV.FLUXER_TOKEN_TYPE ?? 'session';
const BASE = (ENV.FLUXER_API_BASE_URL ?? 'https://api.fluxer.app/v1').replace(/\/+$/, '');

// The 1bit.MONSTER guild + its Text Channels category (from the okf reference).
const GUILD_ID = ENV.FLUXER_HOME_GUILD_ID ?? '1540334785874886656';
const TEXT_CATEGORY_ID = '1540334785874886657';
const SUPPORT_NAME = 'support';
const SUPPORT_TOPIC = 'Official 1bit.MONSTER support. Ask here, get answers from the engine team.';

const WELCOME =
  'Welcome to 1bit.MONSTER support. One engine, any model, zero Python.\n' +
  'Post a question and an engineer (or a bot) will pick it up. Before you ask:\n' +
  '• Check the docs: https://1bit.monster/docs\n' +
  '• Known issue numbers: https://github.com/1bit-MONSTER/1bit-MONSTER/issues\n' +
  'Great first message = what you ran, what you expected, what happened.';

if (!TOKEN) {
  console.error('FLUXER_TOKEN is required (set it in .env or the environment).');
  process.exit(1);
}

const AUTH =
  TOKEN_TYPE === 'bot' ? `Bot ${TOKEN}` : TOKEN_TYPE === 'bearer' ? `Bearer ${TOKEN}` : TOKEN;

async function api(method, path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: {
      Authorization: AUTH,
      'Content-Type': 'application/json',
      'User-Agent': 'fluxer-add-support',
      Accept: 'application/json',
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let data = null;
  if (text) {
    try { data = JSON.parse(text); } catch { data = text; }
  }
  if (!res.ok) {
    // Never embed the token or IDs in errors: log method + status + a short
    // response snippet only (the request path can carry env-derived IDs).
    const snippet = String(data ?? '').replace(/\s+/g, ' ').slice(0, 200);
    throw new Error(`${method} -> HTTP ${res.status}: ${snippet}`);
  }
  return data;
}

async function main() {
  const fresh = process.argv.includes('--fresh');
  console.log(`Preparing #${SUPPORT_NAME} ...`);

  const channels = await api('GET', `/guilds/${GUILD_ID}/channels`);
  let support = channels.find((c) => c.name.toLowerCase() === SUPPORT_NAME && c.type === 0);

  if (support && fresh) {
    await api('DELETE', `/channels/${support.id}`);
    support = undefined;
    console.log(`• deleted existing #${SUPPORT_NAME}`);
  }

  if (!support) {
    support = await api('POST', `/guilds/${GUILD_ID}/channels`, {
      name: SUPPORT_NAME,
      type: 0,
      topic: SUPPORT_TOPIC,
      parent_id: TEXT_CATEGORY_ID,
    });
    console.log(`✔ created #${SUPPORT_NAME} (${support.id})`);
  } else {
    console.log(`• channel #${SUPPORT_NAME} already exists (${support.id})`);
  }

  // Seed a welcome message; skip if we'd be adding to an existing populated thread.
  const messages = await api('GET', `/channels/${support.id}/messages?limit=4`);
  if (!Array.isArray(messages) || messages.length === 0) {
    await api('POST', `/channels/${support.id}/messages`, { content: WELCOME });
    console.log(`✔ seeded welcome message`);
  } else {
    console.log(`• channel already has messages; not re-seeding`);
  }

  console.log(`\nDone. #${SUPPORT_NAME} is ready.`);
}

main().catch((e) => {
  console.error('Failed:', e.message);
  process.exit(1);
});
