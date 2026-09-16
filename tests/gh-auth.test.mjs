// Harness for the gh-auth Pages Function. Runs the real handler against the real
// GitHub API — no mocks — so the check that caught the PKCS#1 import bug runs on
// every change. Exits 0 with a notice when the App secrets are absent (forks).
//
// Usage: APP_ID=… APP_PRIVATE_KEY=… AUTH_TOKEN=… node tests/gh-auth.test.mjs
import { copyFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
// node reads a bare .js in this tree as CJS, so import a copy with an .mjs name.
const tmp = mkdtempSync(join(tmpdir(), 'gh-auth-test-'));
copyFileSync(join(here, '..', 'site', 'functions', 'gh-auth.js'), join(tmp, 'gh-auth.mjs'));
const { onRequest } = await import(join(tmp, 'gh-auth.mjs'));

const env = {
  APP_ID: process.env.APP_ID,
  APP_PRIVATE_KEY: process.env.APP_PRIVATE_KEY,
  AUTH_TOKEN: process.env.AUTH_TOKEN,
  ALLOWED_LOGIN: process.env.ALLOWED_LOGIN || '1bit-MONSTER',
};
if (!env.APP_ID || !env.APP_PRIVATE_KEY || !env.AUTH_TOKEN) {
  console.log('SKIP: APP_ID / APP_PRIVATE_KEY / AUTH_TOKEN not set (fork or unconfigured repo)');
  process.exit(0);
}

const call = async (path, headers = {}) => {
  const res = await onRequest({ request: new Request(`https://1bit.monster${path}`, { headers }), env });
  return { status: res.status, body: await res.text() };
};
const auth = { authorization: `Bearer ${env.AUTH_TOKEN}` };

let failed = 0;
const check = (label, ok, detail = '') => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failed++;
};

const good = await call('/gh-auth', auth);
let minted = null;
try { minted = JSON.parse(good.body); } catch { /* reported below */ }
check('mints with the right bearer', good.status === 200, `status=${good.status}`);
check('  token is an installation token', Boolean(minted?.token?.startsWith('ghs_')), `prefix=${minted?.token?.slice(0, 4)}`);
check('  carries expiry + account', Boolean(minted?.expires_at) && minted?.account === '1bit-MONSTER',
  `expires_in=${minted?.expires_in} account=${minted?.account} installation=${minted?.installation_id}`);
check('  never echoes the key', !good.body.includes('PRIVATE KEY'));

const again = await call('/gh-auth', auth);
check('reuses the cached token', JSON.parse(again.body).token === minted?.token);

check('rejects a wrong bearer', (await call('/gh-auth', { authorization: 'Bearer nope' })).status === 401);
check('rejects a missing bearer', (await call('/gh-auth')).status === 401);
check('rejects other methods', (await call('/gh-auth', { ...auth }) && true) && (await onRequest({
  request: new Request('https://1bit.monster/gh-auth', { method: 'DELETE', headers: auth }), env,
})).status === 405);

const exp = await call('/gh-auth?export=1', auth);
check('?export=1 is eval-friendly', exp.body.startsWith('export GH_TOKEN='), exp.body.slice(0, 18) + '…');

const health = await call('/gh-auth/healthz');
check('healthz needs no auth and leaks nothing', health.status === 200 && !health.body.includes('ghs_'));

console.log(failed === 0 ? '  == 10/10 checks passed ==' : `  == ${failed} FAILED ==`);
process.exit(failed === 0 ? 0 : 1);
