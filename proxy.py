#!/usr/bin/env python3
"""
deepseek-reasoning-proxy — a minimal Anthropic-Messages -> OpenAI shim for
running DeepSeek-V4 (flash/pro) via OpenRouter behind Anthropic-format clients
(Claude Code, etc.) without the thinking-mode `reasoning_content` 400.

The bug
-------
DeepSeek-V4 enables thinking mode by default, and per
https://api-docs.deepseek.com/guides/thinking_mode :

    "Between two user messages, if the model performed a tool call, the
     intermediate assistant's reasoning_content must participate in the context
     concatenation and must be passed back to the API in all subsequent user
     interaction turns."

Anthropic-format clients speak `POST /v1/messages`, and OpenRouter's Anthropic
endpoint does not carry `reasoning_content`, so the first tool round-trip 400s:

    "The `reasoning_content` in the thinking mode must be passed back to the API."

Plain chat works (no tool call -> no requirement); the first tool round-trip
breaks. Setting `reasoning.exclude` makes it worse (it strips the field that
must be echoed); disabling thinking is ignored by the thinking-only model.

The fix (measured)
------------------
On OpenRouter's *OpenAI* endpoint (`/v1/chat/completions`,
`provider {only:[deepseek]}`), an *empty string* `reasoning_content: ""` on the
assistant tool-call message satisfies DeepSeek — you don't have to echo the real
chain-of-thought. This proxy translates Anthropic -> OpenAI, injects that field,
and translates the response back. Native-provider prompt caching is preserved.

Stdlib only. Localhost, single-user. Not hardened for public network exposure.

Env
---
  OPENROUTER_API_KEY      (required)  forwarded as Bearer to the upstream
  PROXY_MODEL             default deepseek/deepseek-v4-flash   (forces the model)
  PROXY_PROVIDER_ONLY     default deepseek    (provider {only:[...]}; empty = no pin)
  PROXY_PORT              default 11456
  PROXY_UPSTREAM          default https://openrouter.ai/api/v1/chat/completions
  PROXY_STREAM_UPSTREAM   default 0   (1 = true token-by-token streaming; 0 =
                          non-streaming upstream + synthesized SSE, which keeps
                          accurate usage + cache-read telemetry)
"""
import os
import sys
import json
import uuid
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("PROXY_UPSTREAM", "https://openrouter.ai/api/v1/chat/completions")
MODEL = os.environ.get("PROXY_MODEL", "deepseek/deepseek-v4-flash")
ONLY = os.environ.get("PROXY_PROVIDER_ONLY", "deepseek")
KEY = os.environ.get("OPENROUTER_API_KEY", "")  # NOT a fallback to inbound client auth:
# clients send a dummy Authorization to the proxy; falling back to it would mask a
# missing OPENROUTER_API_KEY (server starts "OK" but every upstream call 401s).
PORT = int(os.environ.get("PROXY_PORT", "11456"))
STREAM_UPSTREAM = os.environ.get("PROXY_STREAM_UPSTREAM", "0") not in ("0", "", "false", "no")

STOP_MAP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens",
            "content_filter": "end_turn", "function_call": "tool_use"}


def log(*a):
    print("[ds-proxy]", *a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Request translation: Anthropic /v1/messages -> OpenAI chat/completions
# --------------------------------------------------------------------------- #
def _image_part(blk):
    """Anthropic image block -> OpenAI image_url part (base64 or url source)."""
    src = blk.get("source", {})
    if src.get("type") == "base64":
        url = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
    else:
        url = src.get("url", "")
    return {"type": "image_url", "image_url": {"url": url}}


def _user_content(text_parts, image_parts):
    """A string when text-only, else an OpenAI multimodal parts array."""
    if not image_parts:
        return "".join(text_parts)
    parts = []
    joined = "".join(text_parts)
    if joined:
        parts.append({"type": "text", "text": joined})
    parts.extend(image_parts)
    return parts


def to_openai(body):
    msgs = []
    sysp = body.get("system")
    if isinstance(sysp, list):
        sysp = "".join(b.get("text", "") for b in sysp if b.get("type") == "text")
    if sysp:
        msgs.append({"role": "system", "content": sysp})

    for m in body.get("messages", []):
        role = m["role"]
        content = m.get("content", "")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue

        text_parts, image_parts, tool_calls, tool_results = [], [], [], []
        for blk in content:
            t = blk.get("type")
            if t == "text":
                text_parts.append(blk.get("text", ""))
            elif t == "image":
                image_parts.append(_image_part(blk))
            elif t == "tool_use":
                tool_calls.append({
                    "id": blk["id"], "type": "function",
                    "function": {"name": blk["name"],
                                 "arguments": json.dumps(blk.get("input", {}))},
                })
            elif t == "tool_result":
                rc = blk.get("content", "")
                if isinstance(rc, list):
                    rc = "".join(c.get("text", "") for c in rc if c.get("type") == "text")
                tool_results.append({"role": "tool", "tool_call_id": blk["tool_use_id"],
                                     "content": rc if isinstance(rc, str) else json.dumps(rc)})
            # thinking / redacted_thinking are intentionally dropped (the injected
            # reasoning_content:"" is what DeepSeek actually requires).

        if role == "user":
            # Tool responses MUST immediately follow the assistant's tool_calls
            # message (OpenAI/DeepSeek constraint). Any user text in the same turn
            # — e.g. an injected <system-reminder> alongside a tool_result — goes
            # AFTER them, never between the tool_calls message and its responses.
            msgs.extend(tool_results)
            uc = _user_content(text_parts, image_parts)
            if uc:
                msgs.append({"role": "user", "content": uc})
        else:  # assistant
            text = "".join(text_parts)
            if tool_calls:
                msgs.append({"role": "assistant", "content": text or None,
                             "tool_calls": tool_calls,
                             "reasoning_content": ""})  # <-- the fix
            else:
                # never null without tool_calls (DeepSeek rejects assistant null content)
                msgs.append({"role": "assistant", "content": text})

    out = {"model": MODEL, "messages": msgs,
           "max_tokens": body.get("max_tokens", 4096)}
    if ONLY:
        out["provider"] = {"only": [ONLY], "allow_fallbacks": False}
    for k in ("temperature", "top_p"):
        if k in body:
            out[k] = body[k]
    if "stop_sequences" in body:
        out["stop"] = body["stop_sequences"]
    if body.get("tools"):
        out["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object"})}}
            for t in body["tools"]]
    # DeepSeek thinking mode rejects forced tool_choice ("Thinking mode does not
    # support this tool_choice"); only "auto" is honored.
    if body.get("tool_choice"):
        out["tool_choice"] = "auto"
    return out


# --------------------------------------------------------------------------- #
# Response translation: OpenAI -> Anthropic (blocks, one-shot SSE, live SSE)
# --------------------------------------------------------------------------- #
def oai_blocks(msg):
    blocks = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except Exception:
            inp = {}
        blocks.append({"type": "tool_use", "id": tc.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                       "name": fn["name"], "input": inp})
    return blocks


def anthropic_json(oai, model):
    ch = oai["choices"][0]
    usage = oai.get("usage", {}) or {}
    return {"id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant",
            "model": model, "content": oai_blocks(ch["message"]),
            "stop_reason": STOP_MAP.get(ch.get("finish_reason"), "end_turn"),
            "stop_sequence": None,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                      "output_tokens": usage.get("completion_tokens", 0)}}


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _message_start(model, usage):
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return sse("message_start", {"type": "message_start", "message": {
        "id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant",
        "model": model, "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": 0,
                  "cache_read_input_tokens": cached}}})


def synth_sse(oai, model):
    """One-shot: turn a full OpenAI response into a complete Anthropic SSE stream."""
    ch = oai["choices"][0]
    msg = ch["message"]
    usage = oai.get("usage", {}) or {}
    yield _message_start(model, usage)
    idx = 0
    if msg.get("content"):
        yield sse("content_block_start", {"type": "content_block_start", "index": idx,
                  "content_block": {"type": "text", "text": ""}})
        yield sse("content_block_delta", {"type": "content_block_delta", "index": idx,
                  "delta": {"type": "text_delta", "text": msg["content"]}})
        yield sse("content_block_stop", {"type": "content_block_stop", "index": idx})
        idx += 1
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        yield sse("content_block_start", {"type": "content_block_start", "index": idx,
                  "content_block": {"type": "tool_use",
                                    "id": tc.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                                    "name": fn["name"], "input": {}}})
        yield sse("content_block_delta", {"type": "content_block_delta", "index": idx,
                  "delta": {"type": "input_json_delta", "partial_json": fn.get("arguments") or "{}"}})
        yield sse("content_block_stop", {"type": "content_block_stop", "index": idx})
        idx += 1
    yield sse("message_delta", {"type": "message_delta",
              "delta": {"stop_reason": STOP_MAP.get(ch.get("finish_reason"), "end_turn"),
                        "stop_sequence": None},
              "usage": {"output_tokens": usage.get("completion_tokens", 0)}})
    yield sse("message_stop", {"type": "message_stop"})


def _iter_data(resp):
    """Yield parsed JSON objects from an OpenAI-style `data: {...}` SSE stream."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except Exception:
            continue


def stream_sse(resp, model):
    """Live: translate an OpenAI streaming response into Anthropic SSE incrementally.

    Anthropic streams one content block at a time (start -> deltas -> stop). We
    track the currently-open block and close it before opening the next.
    """
    yield _message_start(model, {})
    open_block = None           # ("text", a_idx) | ("tool", a_idx, oai_idx) | None
    next_index = 0
    tool_seen = {}              # oai tool index -> anthropic block index
    finish = None
    usage = {}

    def close():
        nonlocal open_block
        if open_block is not None:
            out = sse("content_block_stop", {"type": "content_block_stop", "index": open_block[1]})
            open_block = None
            return out
        return None

    for chunk in _iter_data(resp):
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        ch0 = choices[0]
        delta = ch0.get("delta") or {}
        if ch0.get("finish_reason"):
            finish = ch0["finish_reason"]

        text = delta.get("content")
        if text:
            if not (open_block and open_block[0] == "text"):
                c = close()
                if c:
                    yield c
                open_block = ("text", next_index)
                yield sse("content_block_start", {"type": "content_block_start",
                          "index": next_index, "content_block": {"type": "text", "text": ""}})
                next_index += 1
            yield sse("content_block_delta", {"type": "content_block_delta",
                      "index": open_block[1], "delta": {"type": "text_delta", "text": text}})

        for tc in delta.get("tool_calls") or []:
            oai_idx = tc.get("index", 0)
            if oai_idx not in tool_seen:
                c = close()
                if c:
                    yield c
                a_idx = next_index
                next_index += 1
                tool_seen[oai_idx] = a_idx
                open_block = ("tool", a_idx, oai_idx)
                fn = tc.get("function") or {}
                yield sse("content_block_start", {"type": "content_block_start", "index": a_idx,
                          "content_block": {"type": "tool_use",
                                            "id": tc.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                                            "name": fn.get("name", ""), "input": {}}})
            args = (tc.get("function") or {}).get("arguments")
            if args:
                yield sse("content_block_delta", {"type": "content_block_delta",
                          "index": tool_seen[oai_idx],
                          "delta": {"type": "input_json_delta", "partial_json": args}})

    c = close()
    if c:
        yield c
    yield sse("message_delta", {"type": "message_delta",
              "delta": {"stop_reason": STOP_MAP.get(finish, "end_turn"), "stop_sequence": None},
              "usage": {"output_tokens": usage.get("completion_tokens", 0)}})
    yield sse("message_stop", {"type": "message_stop"})


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj, close=False):
        b = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            log("client disconnected before response (ignored)")

    def _err(self, code, msg):
        return self._json(code, {"type": "error", "error": {"type": "api_error", "message": msg}})

    def _open_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_POST(self):
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        except (ValueError, TypeError):
            return self._err(400, "bad or missing Content-Length")
        try:
            body = json.loads(raw)
        except Exception:
            return self._json(400, {"type": "error",
                                    "error": {"type": "invalid_request_error", "message": "bad json"}})

        if self.path.endswith("/count_tokens"):
            return self._json(200, {"input_tokens": max(1, len(raw) // 4)})

        try:
            oai_req = to_openai(body)
        except Exception as e:
            log("translate-exc", repr(e))
            return self._json(400, {"type": "error",
                                    "error": {"type": "invalid_request_error",
                                              "message": f"request translation failed: {e!r}"}})

        want_stream = bool(body.get("stream"))
        model = body.get("model", MODEL)
        live = want_stream and STREAM_UPSTREAM
        if live:
            oai_req["stream"] = True
            oai_req["stream_options"] = {"include_usage": True}
        else:
            oai_req["stream"] = False

        req = urllib.request.Request(
            UPSTREAM, data=json.dumps(oai_req).encode(),
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            log("upstream", e.code, txt[:200])
            return self._err(e.code, txt[:600])
        except Exception as e:
            log("upstream-exc", repr(e))
            return self._err(502, repr(e))

        # Live streaming path: translate upstream SSE incrementally.
        if live:
            try:
                self._open_stream()
                for chunk in stream_sse(resp, model):
                    self.wfile.write(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                log("client disconnected mid-stream (ignored)")
            finally:
                resp.close()
            return

        # Buffered path: full response -> accurate usage -> JSON or synthesized SSE.
        try:
            oai = json.loads(resp.read())
        except Exception as e:
            return self._err(502, f"bad upstream JSON: {e!r}")
        finally:
            resp.close()

        if not oai.get("choices"):
            log("no-choices", json.dumps(oai)[:200])
            return self._err(502, f"upstream returned no choices: {json.dumps(oai)[:400]}")

        try:
            if not want_stream:
                return self._json(200, anthropic_json(oai, model))
            chunks = list(synth_sse(oai, model))
        except Exception as e:
            log("synth-exc", repr(e))
            return self._err(502, f"response synthesis failed: {e!r}")

        try:
            self._open_stream()
            for chunk in chunks:
                self.wfile.write(chunk)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log("client disconnected mid-stream (ignored)")


def main():
    if not KEY:
        log("FATAL: set OPENROUTER_API_KEY")
        sys.exit(1)
    mode = "live-stream" if STREAM_UPSTREAM else "buffered (accurate usage)"
    log(f"listening on 127.0.0.1:{PORT}  model={MODEL}  provider_only={ONLY or '(none)'}  mode={mode}")
    log(f"point your client at it:  export ANTHROPIC_BASE_URL=http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
