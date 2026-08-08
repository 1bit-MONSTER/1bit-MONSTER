// ws_session_smoke.js — end-to-end protocol smoke test for the Jarvis
// /v1/voice/session WebSocket path (M1 server work).
//
// Usage: node scripts/ws_session_smoke.js ws://127.0.0.1:8082/v1/voice/session /tmp/jarvis_fixture.pcm16 [token]
//
// Expects: meta + state frames arrive, then either a real transcript/audio
// exchange or (on a box without whisper/LLM/TTS models) an error frame and
// an end frame after the client's stop.  Exits 0 iff an end frame arrived
// (ended=true); exits 1 on socket error; exits 2 on timeout.
//
// Full voice E2E (real Whisper/LLM/TTS) is a manual step on the Strix Halo
// box: start the server there with WHISPER_MODEL_PATH + voice packs, then
// run this script and listen for audio frames.
const fs = require('fs');
const wsUrl = process.argv[2];
const fixture = process.argv[3];
const token = process.argv[4] || '';
if (!wsUrl || !fixture) {
  console.error('usage: node ws_session_smoke.js ws://host:port/v1/voice/session fixture.pcm16 [token]');
  process.exit(2);
}
const pcm = fs.readFileSync(fixture);

const ws = new WebSocket(wsUrl, { headers: token ? { Authorization: 'Bearer ' + token } : {} });
let gotMeta = false, gotState = 0, gotTranscript = 0, gotAudio = 0, ended = false;

ws.onmessage = (ev) => {
  if (typeof ev.data === 'string') {
    const m = JSON.parse(ev.data);
    if (m.type === 'meta' && m.session) gotMeta = true;
    if (m.type === 'state') gotState++;
    if (m.type === 'transcript') { gotTranscript++; console.log(`  transcript[${m.role}]: ${m.text}`); }
    if (m.type === 'end') { ended = true; console.log(`  end reason=${m.reason}`); clearTimeout(timeout); ws.close(); }
    if (m.type === 'error') console.log(`  ERROR: ${m.message}`);
  } else {
    gotAudio += ev.data.byteLength;
  }
};

ws.onopen = () => {
  ws.send(JSON.stringify({ type: 'start' }));
  // stream fixture in 640-byte frames (20 ms @ 16k), 10 ms apart
  let off = 0;
  const timer = setInterval(() => {
    if (off >= pcm.length) { clearInterval(timer); setTimeout(() => ws.send(JSON.stringify({ type: 'stop' })), 800); return; }
    const frame = pcm.subarray(off, off + 640);
    ws.send(frame.buffer.slice(frame.byteOffset, frame.byteOffset + frame.byteLength));
    off += 640;
  }, 10);
  timeout = setTimeout(() => { console.log(`SMOKE TIMEOUT: meta=${gotMeta} states=${gotState} transcripts=${gotTranscript} audioBytes=${gotAudio} ended=${ended}`); process.exit(2); }, 30000);
};

ws.onerror = (e) => { console.log('WS ERROR', e.message || e); process.exit(1); };

ws.onclose = () => { console.log(`SMOKE RESULT: meta=${gotMeta} states=${gotState} transcripts=${gotTranscript} audioBytes=${gotAudio} ended=${ended}`); process.exit(ended ? 0 : 2); };

let timeout = null; // assigned in onopen; declared here so onmessage can clear it
