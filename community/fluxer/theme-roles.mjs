// Applies the site brand palette to the 1bit.MONSTER guild roles via Fluxer.
// Dry-run by default: pass `--apply` to actually PATCH.
//   node scripts/theme-roles.mjs            # dry run (prints what it would do)
//   node scripts/theme-roles.mjs --apply    # write the colors
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

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
const GUILD_ID = ENV.FLUXER_HOME_GUILD_ID ?? '1540334785874886656';
if (!TOKEN) { console.error('FLUXER_TOKEN required'); process.exit(1); }
const AUTH = TOKEN_TYPE === 'bot' ? `Bot ${TOKEN}` : TOKEN_TYPE === 'bearer' ? `Bearer ${TOKEN}` : TOKEN;

const apply = process.argv.includes('--apply');

async function api(method, path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { Authorization: AUTH, 'Content-Type': 'application/json', Accept: 'application/json', 'User-Agent': 'fluxer-theme-roles' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let data = null;
  if (text) { try { data = JSON.parse(text); } catch { data = text; } }
  if (!res.ok) throw new Error(`${method} ${path} -> HTTP ${res.status}: ${JSON.stringify(data)}`);
  return data;
}

// Site palette (hex -> Fluxer int). accent & status from the locked oklch tokens.
const PALETTE = {
  'Admin':       { color: 0x1779e1, label: 'accent blue #1779e1' },  // --accent
  'Moderator':   { color: 0x008048, label: 'status green #008048' },  // --status
  '@everyone':   { color: 0x0e1217, label: 'ink #0e1217' },           // --fg
};

const roles = await api('GET', `/guilds/${GUILD_ID}/roles`);
let changed = 0;
for (const { id, name, color } of roles) {
  const target = PALETTE[name];
  if (!target) continue;
  if ((color ?? 0) === target.color) {
    console.log(`• ${name} already ${target.label}`);
    continue;
  }
  if (!apply) {
    console.log(`dry-run: would set ${name} -> ${target.label} (was color=${color ?? 0})`);
    continue;
  }
  await api('PATCH', `/guilds/${GUILD_ID}/roles/${id}`, { color: target.color });
  console.log(`✔ ${name} -> ${target.label}`);
  changed++;
}
console.log(apply ? `\nApplied ${changed} role color(s).` : `\nDry run only. Run with --apply to write.`);
