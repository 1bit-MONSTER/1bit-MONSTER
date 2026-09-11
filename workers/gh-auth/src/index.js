// gh-auth worker — mints short-lived GitHub App installation tokens over HTTP.
//
// Why: the App private key lives in one place (GitHub Actions secrets / the
// strixhalo helper). A box that needs a token should be able to fetch a
// short-lived one over a link instead of holding the key itself.
//
// Security model (this endpoint hands out credentials — treat it accordingly):
//   * REQUIRES `Authorization: Bearer <AUTH_TOKEN>`; without it, 401 and nothing
//     is minted. Put Cloudflare Access in front of it for a second factor.
//   * the App private key never leaves Worker secrets; it is never returned or logged
//   * tokens are installation tokens: 1 hour, revocable, attributed to the App —
//     NOT to a user, so they cannot do anything the installation cannot do
//   * responses are `no-store`; no request logging of the Authorization header
//
// Routes:
//   GET  /            -> {"token","expires_at","account","installation_id","expires_in"}
//   GET  /?export=1   -> text/plain `export GH_TOKEN='...'`  (eval-friendly)
//   GET  /healthz     -> {"ok":true,...} — no auth, no secrets, for readiness checks
//
// Required secrets: APP_ID, APP_PRIVATE_KEY (PKCS#8 PEM), AUTH_TOKEN
// Optional vars:    ALLOWED_LOGIN (default "1bit-MONSTER"), APP_INSTALL_ID

const enc = new TextEncoder();

function b64url(bytes) {
  const b = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : bytes;
  let s = "";
  for (const byte of b) s += String.fromCharCode(byte);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
const b64urlText = (s) => b64url(enc.encode(s));

// DER length octets (definite form; keys are > 127 bytes, so long form happens).
function derLen(n) {
  if (n < 0x80) return [n];
  const out = [];
  while (n > 0) { out.unshift(n & 0xff); n >>= 8; }
  return [0x80 | out.length, ...out];
}

// PKCS#8 PrivateKeyInfo envelope around a PKCS#1 RSAPrivateKey:
//   SEQUENCE { INTEGER 0, SEQUENCE { OID rsaEncryption, NULL }, OCTET STRING <pkcs1> }
function wrapPkcs1AsPkcs8(pkcs1) {
  const version = [0x02, 0x01, 0x00];
  const alg = [0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01, 0x05, 0x00];
  const octet = [0x04, ...derLen(pkcs1.length), ...pkcs1];
  const body = [...version, ...alg, ...octet];
  return Uint8Array.from([0x30, ...derLen(body.length), ...body]);
}

// GitHub hands out App keys as "BEGIN RSA PRIVATE KEY" (PKCS#1), which WebCrypto
// cannot import; PKCS#8 is accepted as-is. Handle both here rather than making
// callers convert a secret that other workflows also use.
function pemToDer(pem) {
  const text = String(pem);
  const header = (text.match(/-----BEGIN ([^-]+)-----/) || [])[1] || "";
  const body = text
    .replace(/-----BEGIN [^-]*-----/, "")
    .replace(/-----END [^-]*-----/, "")
    .replace(/\s+/g, "");
  const bin = atob(body);
  const der = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) der[i] = bin.charCodeAt(i);
  return /RSA PRIVATE KEY/.test(header) ? wrapPkcs1AsPkcs8(der) : der;
}

async function appJwt(env, now) {
  const key = await crypto.subtle.importKey(
    "pkcs8", pemToDer(env.APP_PRIVATE_KEY),
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["sign"]);
  const head = b64urlText(JSON.stringify({ alg: "RS256", typ: "JWT" }));
  const body = b64urlText(JSON.stringify({
    iat: now - 60, exp: now + 540, iss: String(env.APP_ID),
  }));
  const data = `${head}.${body}`;
  return `${data}.${b64url(await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, enc.encode(data)))}`;
}

async function gh(path, init) {
  return fetch(`https://api.github.com${path}`, {
    ...init,
    headers: {
      accept: "application/vnd.github+json",
      "user-agent": "1bit-gh-auth-worker",
      "x-github-api-version": "2022-11-28",
      ...(init?.headers || {}),
    },
  });
}

// one token per isolate, reused until it is nearly expired
let cached = null;

async function mintInstallationToken(env, now) {
  if (cached && cached.expiresMs - 60_000 > now) return cached;
  const jwt = await appJwt(env, Math.floor(now / 1000));
  const auth = { authorization: `Bearer ${jwt}` };

  let installId = env.APP_INSTALL_ID;
  let account = env.ALLOWED_LOGIN || "1bit-MONSTER";
  if (!installId) {
    const res = await gh("/app/installations", { headers: auth });
    if (!res.ok) throw new Error(`installations lookup failed: ${res.status}`);
    const list = await res.json();
    const pick = list.find((i) => i.account?.login === account) || list[0];
    if (!pick) throw new Error("the App has no installations");
    installId = pick.id;
    account = pick.account?.login || account;
  }

  const res = await gh(`/app/installations/${installId}/access_tokens`, {
    method: "POST", headers: auth,
  });
  if (!res.ok) throw new Error(`token mint failed: ${res.status}`);
  const body = await res.json();
  cached = {
    token: body.token,
    expiresAt: body.expires_at,
    expiresMs: Date.parse(body.expires_at),
    account,
    installationId: installId,
  };
  return cached;
}

function timingSafeEqual(a, b) {
  const x = enc.encode(String(a ?? ""));
  const y = enc.encode(String(b ?? ""));
  if (x.length !== y.length) return false;
  let diff = 0;
  for (let i = 0; i < x.length; i++) diff |= x[i] ^ y[i];
  return diff === 0;
}

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj, null, 2) + "\n", {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/healthz") {
      return json({
        ok: true,
        configured: Boolean(env.APP_ID && env.APP_PRIVATE_KEY && env.AUTH_TOKEN),
        account: env.ALLOWED_LOGIN || "1bit-MONSTER",
      });
    }
    if (request.method !== "GET" && request.method !== "POST") {
      return json({ error: "method not allowed" }, 405);
    }
    if (!env.AUTH_TOKEN) {
      return json({ error: "worker is not configured (AUTH_TOKEN secret missing)" }, 500);
    }
    const header = request.headers.get("authorization") || "";
    const presented = header.replace(/^Bearer\s+/i, "").trim();
    if (!timingSafeEqual(presented, env.AUTH_TOKEN)) {
      return json({ error: "unauthorized" }, 401);
    }

    try {
      const tok = await mintInstallationToken(env, Date.now());
      const seconds = Math.max(0, Math.floor((tok.expiresMs - Date.now()) / 1000));
      if (url.searchParams.has("export")) {
        return new Response(`export GH_TOKEN='${tok.token}'\n`, {
          headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store" },
        });
      }
      return json({
        token: tok.token,
        expires_at: tok.expiresAt,
        expires_in: seconds,
        account: tok.account,
        installation_id: tok.installationId,
        usage: "GH_TOKEN=<token> gh pr list -R 1bit-MONSTER/1bit-MONSTER",
      });
    } catch (err) {
      // Never echo secrets or the JWT; the message names the failing step only.
      return json({ error: String(err.message || err) }, 502);
    }
  },
};
