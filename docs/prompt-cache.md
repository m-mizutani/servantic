# Prompt Caching

gollem can observe provider prompt-cache usage across all providers, and — for
Claude — opt into writing cacheable breakpoints so repeated prompt prefixes are
served from the provider's cache at a lower cost.

There are two independent capabilities:

1. **Observation (always on).** Every response reports how many input tokens were
   served from the prompt cache, for all providers. No configuration required.
2. **Claude write path (opt-in).** When enabled, gollem marks the stable prefix
   (system prompt and tools) and the growing conversation tail with ephemeral
   `cache_control`, so Claude caches them.

## Observing cache usage

`gollem.Response` (and the streaming responses) carry a per-call token breakdown:

```go
type Response struct {
    // ...
    InputToken              int // total input tokens (includes cached reads)
    OutputToken             int
    CacheCreationInputToken int // input tokens written to the cache (Claude only)
    CacheReadInputToken     int // input tokens served from the cache (cache hits)
}
```

- `InputToken` always means **total input** — the tokens after the cache
  breakpoint plus the cached prefix. This keeps token-budget logic (such as the
  compacter) correct whether or not caching is active.
- `CacheReadInputToken` is the portion of the input that was a cache hit.
- `CacheCreationInputToken` is the portion written to the cache on this call.
  Only Claude distinguishes cache writes; it is `0` for OpenAI and Gemini.

Example:

```go
resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("...")})
if err != nil {
    return err
}
log.Printf("input=%d (cached read=%d, cache write=%d) output=%d",
    resp.InputToken, resp.CacheReadInputToken,
    resp.CacheCreationInputToken, resp.OutputToken)
```

The same values are recorded on trace spans through
`trace.LLMCallData.CacheCreationInputTokens` / `CacheReadInputTokens`
(`trace/logger` emits `cache_creation_input_tokens` / `cache_read_input_tokens`;
`trace/otel` sets `llm.cache_creation_input_tokens` /
`llm.cache_read_input_tokens`). They are omitted when zero. The same provider
asymmetry applies: `CacheCreationInputTokens` is reported by Claude only, so `0`
on OpenAI or Gemini does not mean the cache missed.

## Enabling Claude prompt caching

Enable it at the agent level:

```go
agent := gollem.New(client, gollem.WithPromptCache(true))
```

or on a standalone session:

```go
session, err := client.NewSession(ctx, gollem.WithSessionPromptCache(true))
```

or on a one-shot structured query:

```go
resp, err := gollem.Query[Answer](ctx, client, prompt,
    gollem.WithQuerySystemPrompt(sharedSystemPrompt),
    gollem.WithQueryPromptCache(true),
)
```

A single query rarely hits the cache on its own; this pays off for recurring
queries that share the same system prompt.

All three default to disabled. When enabled, gollem adds `cache_control` breakpoints
to the Claude request in up to three places:

- the **system prompt** (last block),
- the **tools** (last tool), and
- the **conversation tail** (last content block of the last message).

The system prompt and tools are stable across turns, so they become cache hits
after the first call. The conversation-tail breakpoint moves forward as the
conversation grows, so each new turn only pays full price for the newly added
tokens — this is the biggest win for long agentic loops.

Enabling caching only changes Claude requests. OpenAI and Gemini cache
automatically on the provider side, so the flag does not alter their requests;
their cache usage is still reported through the observation fields above.

## Provider support

| Provider | Observation | Write control | Notes |
|----------|-------------|---------------|-------|
| Claude   | ✅ read + write | ✅ auto breakpoints (`WithPromptCache`) | System, tools, and conversation tail are marked when enabled. |
| OpenAI   | ✅ read | — (automatic) | Caches automatically for long prompts; `CacheReadInputToken` reflects `prompt_tokens_details.cached_tokens`. |
| Gemini   | ✅ read | — (implicit) | Caches implicitly for repeated prefixes; `CacheReadInputToken` reflects `cachedContentTokenCount`. Hit rate is model-dependent (see limitations). |

## Notes and limitations

- **Minimum cacheable length.** Claude only caches prefixes above a
  model-specific minimum (for example 1024 tokens for some models, 4096 for
  others). Marking shorter content is a no-op on the API side — no error is
  returned, and the cache token counts simply stay `0`.
- **Growing conversations.** Claude checks a limited look-back window for a
  previously cached prefix. If a single turn appends a very large number of
  content blocks, an incremental cache hit at the tail may be missed. Typical
  agentic turns (a message and a tool result) stay well within the window.
- **TTL.** Claude cache entries use the default 5-minute TTL, which is not
  configurable through gollem.
- **Gemini implicit caching is model-dependent.** Gemini caches automatically
  (implicitly) for large repeated prefixes and reports the hit through
  `CacheReadInputToken`, with none of the extra API surface Claude needs. The
  hit rate depends on the model, though: it is reliable on `gemini-2.5-flash`,
  flaky on `gemini-3.5-flash`, and did not fire on `gemini-2.5-flash-lite` in
  testing. gollem exposes no client-side Gemini cache control.
- **Claude explicit breakpoints.** Manually choosing Claude cache positions in
  the middle of a conversation is not currently exposed; the automatic
  breakpoints above cover the common cases.
