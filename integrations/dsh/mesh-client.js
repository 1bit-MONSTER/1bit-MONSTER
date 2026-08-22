// mesh-client.js — 1bit-MONSTER Mesh API client (plain Node 18+, no deps).
//
// Tiny HTTP client for a mesh node's /v1/mesh/* surface. Used by the DSH
// brain (mesh-brain.js) and by DSH agents that want to drive a node.
//
// Usage:
//   const mesh = require('./mesh-client');
//   const n = new mesh.MeshNode('http://192.168.1.20:8088');
//   console.log(await n.me());
//   console.log(await n.peers());
//   await n.ask('peer-id', 'want to hook up and integrate?');
'use strict';

class MeshNode {
  /**
   * @param {string} baseUrl node HTTP base, e.g. 'http://host:8088'
   * @param {object} [opts]
   * @param {number} [opts.timeoutMs=5000]
   */
  constructor(baseUrl, opts = {}) {
    let u = String(baseUrl).replace(/\/+$/, '');
    // A bare origin (http://host:port) is the node root → api_base is /v1.
    // A URL that already carries a path is used as-is.
    if (/^https?:\/\/[^/]+$/.test(u)) u += '/v1';
    this.baseUrl = u;
    this.timeoutMs = opts.timeoutMs ?? 5000;
  }

  async _request(method, path, body) {
    return this._requestOn(this.baseUrl, method, path, body);
  }

  async _requestOn(baseUrl, method, path, body) {
    // api_base may carry a path (e.g. 'http://host:8088/v1'); the request
    // path is then relative to it — mirror the C++ client's normalization.
    let base = String(baseUrl).replace(/\/+$/, '');
    let basePath = '';
    const m = /^(https?:\/\/[^/]+)(\/.*)?$/.exec(base);
    if (m) {
      base = m[1];
      basePath = m[2] ?? '';
    }
    const url = base + basePath + path;
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), this.timeoutMs);
    try {
      const res = await fetch(url, {
        method,
        headers: body ? { 'content-type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        signal: ctl.signal,
      });
      if (!res.ok) throw new Error(`${method} ${url} -> HTTP ${res.status}: ${await res.text()}`);
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  /** GET /v1/mesh/me — this node's identity card. */
  async me() {
    const r = await this._request('GET', '/mesh/me');
    return r.node;
  }

  /** GET /v1/mesh/peers — live sibling installs. */
  async peers() {
    const r = await this._request('GET', '/mesh/peers');
    return r.peers ?? [];
  }

  /** POST /v1/mesh/handshake — capability exchange ("hook up"). */
  async handshake(peerNodeCard) {
    return this._request('POST', '/mesh/handshake', { node: peerNodeCard });
  }

  /**
   * POST /v1/mesh/ask — deliver a question to a peer.
   * The question goes to the TARGET node's server (peer.api_base); the
   * `from` card is this node's.
   * @param {string} targetApiBase  peer's api_base, e.g. 'http://host:8088/v1'
   * @param {string} question       the question text
   * @param {string} [type='question'] 'intro' | 'question' | 'integration_offer'
   */
  async ask(targetApiBase, question, type = 'question', extra = {}) {
    const me = await this.me();
    return this._requestOn(targetApiBase, 'POST', '/mesh/ask', {
      from: me.id,
      from_name: me.name,
      ask_id: `dsh-${Date.now()}-${Math.floor(Math.random() * 1e6)}`,
      type,
      question,
      node: me,
      ...extra,
    });
  }

  /** POST /v1/mesh/answer — reply to an ask on the ASKER's server (accept integration). */
  async answer(targetApiBase, askId, answer, accept = true) {
    const me = await this.me();
    return this._requestOn(targetApiBase, 'POST', '/mesh/answer', {
      ask_id: askId,
      from: me.id,
      from_name: me.name,
      answer,
      accept,
    });
  }

  /** GET /v1/mesh/asks — conversation log (inbound + outbound). */
  async asks() {
    const r = await this._request('GET', '/mesh/asks');
    return r.asks ?? [];
  }

  /**
   * Local chat via the node's own OpenAI-compatible endpoint — the fully
   * local "brain" for generating questions / answers. Falls back to null
   * when the node serves no model.
   */
  async localChat(system, user, opts = {}) {
    try {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), (opts.timeoutMs ?? 30000));
      const res = await fetch(this.baseUrl + '/chat/completions', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model: opts.model ?? 'auto',
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: user },
          ],
          max_tokens: opts.maxTokens ?? 220,
          temperature: opts.temperature ?? 0.6,
          stream: false,
        }),
        signal: ctl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) return null;
      const j = await res.json();
      return j?.choices?.[0]?.message?.content?.trim() ?? null;
    } catch {
      return null; // no model loaded on this node → caller falls back to template
    }
  }
}

module.exports = { MeshNode };
