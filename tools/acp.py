#!/usr/bin/env python3
"""ACP client — Agentic Control Protocol for Abacus.AI RouteLLM.

Stdlib-only (urllib). See docs/agentic-control-protocol.md for the spec.

Usage:
  acp.py check                        # verify key + reachability
  acp.py models [FILTER]              # list models (grep filter)
  acp.py chat "prompt"                # auto-routed chat (route-llm)
        --system "sys prompt" --model ID --max-tokens N --temp T
        --json                        # json_object mode, pretty-printed
        --stream                      # SSE stream to stdout
        --show-model                  # print served model id
        --image URL...                # attach image_url parts
        --file URL...                 # attach PDF/file parts (type:file)
  acp.py self-test                    # run against an in-process mock server

Key: $ABACUS_API_KEY > $ACP_API_KEY > ~/.config/abacus/route-llm.key
Base URL: $ACP_BASE_URL (default https://routellm.abacus.ai/v1)
"""
import argparse, json, os, sys, threading, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_BASE = "https://routellm.abacus.ai/v1"


def get_key():
    key = os.environ.get("ABACUS_API_KEY") or os.environ.get("ACP_API_KEY")
    if not key:
        path = os.path.expanduser("~/.config/abacus/route-llm.key")
        if os.path.exists(path):
            key = open(path).read().strip()
    return key


def api(base, key, method, path, body=None, timeout=120):
    req = urllib.request.Request(base.rstrip("/") + path, method=method)
    req.add_header("Authorization", "Bearer " + key)
    if body is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(body).encode()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")


def stream_chat(base, key, body, timeout=120):
    """Yield (delta_text, tool_calls_agg, usage). Accepts a normal request; splits at /chat/completions."""
    path = "/chat/completions"
    req = urllib.request.Request(base.rstrip("/") + path, method="POST")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    req.data = json.dumps({**body, "stream": True}).encode()
    buf, tools = "", {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(f"stream error: {chunk['error']}")
            if chunk.get("usage"):
                yield chunk["usage"], None, None
                continue
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content"):
                    yield d["content"], None, None
                for tc in d.get("tool_calls", []):
                    t = tools.setdefault(tc["index"], {"id": "", "name": "", "args": ""})
                    t["id"] += tc.get("id", "")
                    t["name"] += tc.get("function", {}).get("name", "")
                    t["args"] += tc.get("function", {}).get("arguments", "")
    if tools:
        yield None, tools, None


def do_chat(args, key, base):
    content = []
    for u in args.image or []:
        content.append({"type": "image_url", "image_url": {"url": u}})
    for u in args.file or []:
        content.append({"type": "file", "file": {"filename": u.rsplit("/", 1)[-1], "file_data": u}})
    if content:
        content.insert(0, {"type": "text", "text": args.prompt})
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": args.prompt}]
    if args.system:
        messages.insert(0, {"role": "system", "content": args.system})
    body = {"model": args.model, "messages": messages}
    if args.max_tokens:
        body["max_tokens"] = args.max_tokens
    if args.temp is not None:
        body["temperature"] = args.temp
    if args.json:
        body["response_format"] = {"type": "json_object"}
        if not args.system:
            messages.insert(0, {"role": "system", "content": "Output JSON only."})

    if args.stream:
        usage = None
        for item, tools, u in stream_chat(base, key, body):
            if u:
                usage = u
            elif item:
                sys.stdout.write(item)
                sys.stdout.flush()
        sys.stdout.write("\n")
        if usage:
            print(f"# usage: {usage}", file=sys.stderr)
        return

    _, resp = api(base, key, "POST", "/chat/completions", body)
    msg = resp["choices"][0]["message"]
    if args.show_model:
        print(f"# model: {resp.get('model')}", file=sys.stderr)
    if msg.get("tool_calls"):
        print(json.dumps(msg, indent=2))
    elif args.json:
        print(json.dumps(json.loads(msg["content"]), indent=2))
    else:
        print(msg.get("content") or "")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="verify key + reachability")
    sub.add_parser("self-test", help="run mock-server self test")
    m = sub.add_parser("models", help="list models")
    m.add_argument("filter", nargs="?", default=None)
    c = sub.add_parser("chat", help="chat with auto-route")
    c.add_argument("prompt")
    c.add_argument("--system", default=None)
    c.add_argument("--model", default="route-llm")
    c.add_argument("--max-tokens", type=int, default=None)
    c.add_argument("--temp", type=float, default=None)
    c.add_argument("--json", action="store_true")
    c.add_argument("--stream", action="store_true")
    c.add_argument("--show-model", action="store_true")
    c.add_argument("--image", action="append", default=None)
    c.add_argument("--file", action="append", default=None)
    args = p.parse_args()

    if args.cmd == "self-test":
        sys.exit(self_test())
    base = os.environ.get("ACP_BASE_URL", DEFAULT_BASE)
    key = get_key()
    if args.cmd != "check" and not key:
        print("No API key. Set ABACUS_API_KEY (or ACP_API_KEY, or ~/.config/abacus/route-llm.key).")
        sys.exit(2)
    try:
        if args.cmd == "check":
            if not key:
                print("No API key configured (ABACUS_API_KEY / ACP_API_KEY / ~/.config/abacus/route-llm.key).")
                sys.exit(2)
            _, resp = api(base, key, "GET", "/models")
            ids = [x["id"] for x in resp.get("data", [])]
            print(f"OK — {len(ids)} models. route-llm={'route-llm' in ids} sample: {ids[:5]}")
        elif args.cmd == "models":
            _, resp = api(base, key, "GET", "/models")
            rows = [x for x in resp.get("data", []) if not args.filter or args.filter in str(x)]
            for x in rows:
                print(x.get("id"), "|", x.get("pricing", {}).get("input", "?"), "/", x.get("pricing", {}).get("output", "?"))
        elif args.cmd == "chat":
            do_chat(args, key, base)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


# ---- self-test: in-process mock RouteLLM server ---------------------------

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/v1/models":
            self._json({"data": [{"id": "route-llm", "pricing": {"input": 1, "output": 2}},
                                 {"id": "gpt-5.4", "pricing": {"input": 3, "output": 4}}]})
        else:
            self.send_error(404)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if not req.get("messages"):
            self.send_error(400, "missing messages")
            return
        if req.get("stream"):
            body = req["messages"][-1]["content"]
            chunks = [f'data: {json.dumps({"choices":[{"delta":{"content":c}}]})}\n\n' for c in body] + ["data: [DONE]\n\n"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write("".join(chunks).encode())
            return
        msg = {"role": "assistant", "content": "mock answer to: " + req["messages"][-1]["content"]}
        if req.get("tools"):
            msg = {"role": "assistant", "content": None,
                   "tool_calls": [{"id": "call_1", "type": "function",
                                   "function": {"name": req["tools"][0]["function"]["name"], "arguments": "{}"}}]}
        self._json({"id": "x", "object": "chat.completion", "model": req.get("model", "route-llm"),
                    "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if req.get("tools") else "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def self_test():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    key = "test-key"
    # models
    _, m = api(base, key, "GET", "/models")
    assert any(x["id"] == "route-llm" for x in m["data"]), "models list"
    # plain chat
    _, r = api(base, key, "POST", "/chat/completions",
               {"model": "route-llm", "messages": [{"role": "user", "content": "hi"}]})
    assert r["choices"][0]["message"]["content"].startswith("mock answer"), "chat"
    assert r["model"] == "route-llm", "served model"
    # tool calling shape
    _, r = api(base, key, "POST", "/chat/completions",
               {"model": "route-llm",
                "messages": [{"role": "user", "content": "w"}],
                "tools": [{"type": "function", "function": {"name": "foo", "parameters": {"type": "object", "properties": {}}}}]})
    assert r["choices"][0]["finish_reason"] == "tool_calls", "tool finish_reason"
    assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "foo", "tool name"
    # streaming
    out = "".join(x for x, _, _ in stream_chat(base, key, {"messages": [{"role": "user", "content": "abc"}]}))
    assert out == "abc", f"stream got {out!r}"
    # error surface
    try:
        api(base, key, "POST", "/chat/completions", {"bad": True})
        raise AssertionError("expected HTTP error")
    except RuntimeError:
        pass
    srv.shutdown()
    print("self-test OK")
    return 0


if __name__ == "__main__":
    main()
