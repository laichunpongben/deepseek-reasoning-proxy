# deepseek-reasoning-proxy

A tiny, zero-dependency local proxy that lets **Anthropic-Messages-API clients**
(Claude Code and friends) use **DeepSeek-V4** (flash/pro) via **OpenRouter**
without the thinking-mode `reasoning_content` **400**.

```
Claude Code  ──Anthropic /v1/messages──▶  this proxy  ──OpenAI /chat/completions──▶  OpenRouter ▶ DeepSeek
                                          (injects reasoning_content:"")
```

> **Stopgap, not a framework.** This exists because of a specific DeepSeek-V4
> regression. The upstream clients are fixing it natively (see
> [Prior art](#prior-art)); check whether your client already handles it before
> reaching for this. It's ~1 file of stdlib Python on purpose.

## The bug it fixes

DeepSeek-V4 enables thinking mode by default and, per the
[DeepSeek Thinking Mode docs](https://api-docs.deepseek.com/guides/thinking_mode):

> *"Between two `user` messages, if the model performed a tool call, the
> intermediate `assistant`'s `reasoning_content` must participate in the context
> concatenation and must be **passed back to the API** in all subsequent user
> interaction turns."*

Anthropic-format clients speak `POST /v1/messages`, and OpenRouter's Anthropic
endpoint doesn't carry `reasoning_content`. So the **first tool round-trip** fails:

```
API Error: 400 — The `reasoning_content` in the thinking mode must be passed back to the API.
```

Plain chat works (no tool call → no requirement); the first tool call breaks. And
the usual knobs don't help — `reasoning.exclude` makes it **worse** (it strips the
field that must be echoed), and disabling thinking is ignored by the thinking-only
model.

**The fix (measured):** on OpenRouter's *OpenAI* endpoint, an **empty string**
`reasoning_content: ""` on the assistant tool-call message satisfies DeepSeek —
you don't have to echo the real chain-of-thought. This proxy translates
Anthropic → OpenAI, injects that field, and translates the response back. Native
prompt caching is preserved (the injected `""` keeps the prefix deterministic).

## Install & run

No dependencies — just Python 3.9+.

```bash
git clone https://github.com/laichunpongben/deepseek-reasoning-proxy
cd deepseek-reasoning-proxy

export OPENROUTER_API_KEY=sk-or-...           # your OpenRouter key
python3 proxy.py                               # listens on 127.0.0.1:11434
```

Point your client at it:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:11434"
export ANTHROPIC_AUTH_TOKEN="unused-proxy-ignores-it"   # clients require *a* value; the proxy ignores it
# then run your Anthropic-format client (e.g. Claude Code)
```

That's it. Tool calls now round-trip — DeepSeek-V4 + native cache + tool use.

## Configuration

| Env | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | forwarded as `Bearer` to the upstream |
| `PROXY_MODEL` | `deepseek/deepseek-v4-flash` | model sent upstream (overrides the client's) |
| `PROXY_PROVIDER_ONLY` | `deepseek` | `provider {only:[…]}` pin; set empty to disable |
| `PROXY_PORT` | `11434` | localhost port (note: collides with Ollama's default) |
| `PROXY_UPSTREAM` | `https://openrouter.ai/api/v1/chat/completions` | OpenAI-format endpoint |
| `PROXY_STREAM_UPSTREAM` | `0` | `1` = true token-by-token streaming; `0` = buffered upstream + synthesized SSE (keeps accurate usage + cache-read telemetry) |

### Two streaming modes

- **Buffered (default):** the upstream call is non-streaming; the proxy synthesizes
  the Anthropic SSE stream from the full response. Responses land all at once, but
  **usage and `cache_read_input_tokens` are exact** — best for background/agent loops
  where cost telemetry matters more than typing animation.
- **Live (`PROXY_STREAM_UPSTREAM=1`):** true token-by-token translation. Better
  interactive feel; input-token/cache usage is approximate (OpenAI streaming doesn't
  report input tokens until the end).

## Notes & limitations

- **Use a generous `max_tokens`.** In thinking mode the model spends tokens
  *reasoning* (which this proxy drops from the output) before any answer — a tiny
  `max_tokens` can yield an empty/`length`-truncated reply. Most clients default high.
- **Forced `tool_choice` is downgraded to `auto`** — DeepSeek thinking mode rejects
  `any`/required/specific (`"Thinking mode does not support this tool_choice"`).
- **Localhost, single-user. Not hardened for public network exposure.** It forwards
  your `OPENROUTER_API_KEY` upstream and ignores the client's inbound auth.
- Reasoning/thinking content is dropped from responses (not surfaced as Anthropic
  `thinking` blocks). The empty `reasoning_content` echo is all DeepSeek requires.

## Running it as a service

See [`examples/`](examples/) for a macOS **launchd** plist and a Linux **systemd**
unit so the proxy stays up across reboots/logouts.

## Prior art

This is the same bug tracked across the ecosystem — e.g.
[claude-code-router #1378](https://github.com/musistudio/claude-code-router/issues/1378),
and similar fixes in Kilo Code, OpenCode, Roo, and Copilot Chat. If you already use
a router/proxy, prefer its native fix. This repo is the minimal standalone option
for people who point their client **straight** at OpenRouter.

## Tests

```bash
python3 -m unittest discover -s tests
```
Network-free: translation + a mocked-upstream handler. No API key needed.

## License

[MIT](LICENSE)
