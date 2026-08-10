# LLM Provider Configuration

This document provides detailed configuration options for each LLM provider supported by gollem.

## Table of Contents

- [Gemini](#gemini)
- [Claude (Anthropic)](#claude-anthropic)
- [Claude (Vertex AI)](#claude-vertex-ai)
- [OpenAI](#openai)

## Gemini

### Basic Setup

```go
import (
    "context"
    "github.com/gollem-dev/gollem/llm/gemini"
)

client, err := gemini.New(ctx, "your-project-id", "us-central1")
```

### Authentication

Gemini uses Google Cloud credentials. Set up authentication using one of:

```bash
# Option 1: Service account key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"

# Option 2: gcloud CLI
gcloud auth application-default login

# Option 3: Workload identity (automatic in GKE/Cloud Run)
```

### Configuration Options

#### Model Selection

```go
client, err := gemini.New(ctx, projectID, location,
    gemini.WithModel("gemini-1.5-pro-latest"),
)
```

Available models (default: `gemini-3.5-flash`):
- `gemini-3.5-flash` - Latest Flash model with improved agent execution and coding (uses thinking levels)
- `gemini-2.5-pro` - Advanced model with state-of-the-art thinking capabilities
- `gemini-2.5-flash` - Strong price-performance model with well-rounded capabilities
- `gemini-2.5-flash-lite` - Optimized for cost efficiency and low latency
- `gemini-2.0-flash` - Superior speed with native tool use and 1M token context
- `gemini-2.0-flash-thinking-exp-1219` - Experimental model with thinking capabilities

Note: Gemini 1.5 models are deprecated as of April 2025 for new projects.

#### Thinking Level (Gemini 3.x)

Gemini 3.x replaces the numeric `thinking_budget` with a discrete `thinking_level`:

```go
// Use minimal thinking for fast, simple responses
client, err := gemini.New(ctx, projectID, location,
    gemini.WithModel("gemini-3.5-flash"),
    gemini.WithThinkingLevel(genai.ThinkingLevelMinimal),
)

// Use higher levels for harder reasoning tasks
client, err := gemini.New(ctx, projectID, location,
    gemini.WithModel("gemini-3.5-flash"),
    gemini.WithThinkingLevel(genai.ThinkingLevelHigh),
)
```

Available levels (lowest → highest): `ThinkingLevelMinimal`, `ThinkingLevelLow`, `ThinkingLevelMedium`, `ThinkingLevelHigh`.

Without `WithThinkingLevel` (or `WithThinkingBudget`), gollem sends no thinking configuration at all and each model applies its own default — `MEDIUM` for Gemini 3.5 / 3.6 Flash, `HIGH` for Gemini 3 Pro, `MINIMAL` for Gemini 3.5 Flash Lite. Not every model accepts every level (`gemini-3-pro-preview` takes only `LOW` and `HIGH`), so set the level only when you know the model supports it.

Note: Gemini 3.x also deprecates `temperature`, `top_p`, and `top_k` — omit those options when using 3.x models.

#### Thought Summaries

Gemini returns the model's reasoning text only when thought summaries are requested. Without this option `Response.Thoughts` is always empty:

```go
client, err := gemini.New(ctx, projectID, location,
    gemini.WithIncludeThoughts(true),
)
```

This affects the reasoning text only. The thought signatures that Gemini 3.x requires for multi-turn tool calling are returned and stored in history regardless of this setting, in both blocking and streaming modes.

#### Thinking Budget (Gemini 2.x)

For Gemini 2.x models, control thinking via a numeric token budget:

```go
// Automatic thinking budget (model decides based on complexity)
client, err := gemini.New(ctx, projectID, location,
    gemini.WithThinkingBudget(-1),
)

// Fixed token budget for thinking
client, err := gemini.New(ctx, projectID, location,
    gemini.WithThinkingBudget(1000), // 1000 tokens
)

// Disable thinking
client, err := gemini.New(ctx, projectID, location,
    gemini.WithThinkingBudget(0),
)
```

The thinking budget controls computational effort for internal reasoning:
- **-1**: Automatic mode - the model decides based on task complexity
- **Positive value**: Fixed token budget for thinking
- **0**: Disable thinking mode

This feature is particularly useful for complex reasoning tasks where you want the model to spend more time thinking through problems before responding.

#### Temperature and Other Parameters

```go
client, err := gemini.New(ctx, projectID, location,
    gemini.WithTemperature(0.7),
    gemini.WithMaxTokens(8192),  // Optional, omit for model's max capacity
    gemini.WithTopP(0.9),
)
```

### Environment Variables

- `GEMINI_PROJECT_ID` - Google Cloud project ID
- `GEMINI_LOCATION` - Vertex AI location (e.g., "us-central1")
- `GOLLEM_LOGGING_GEMINI_PROMPT` - Enable prompt logging for debugging
- `GOLLEM_LOGGING_GEMINI_RESPONSE` - Enable response logging for debugging

## Claude (Anthropic)

### Basic Setup

```go
import (
    "context"
    "github.com/gollem-dev/gollem/llm/claude"
)

client, err := claude.New(ctx, "your-api-key")
```

### Configuration Options

#### Model Selection

```go
client, err := claude.New(ctx, apiKey,
    claude.WithModel("claude-sonnet-4-5-20250929"),
)
```

Available models:
- `claude-sonnet-4-5-20250929` - Latest Sonnet 4.5 model (default)
- `claude-opus-4-1-20250805` - Most powerful model, best for complex tasks (August 2025)
- `claude-sonnet-4-20250514` - Balanced performance and efficiency
- `claude-3-5-sonnet-20241022` - Previous generation, still widely available
- `claude-3-5-haiku-20241022` - Fast, cost-effective model

Note: Claude Opus 4.1 and Sonnet 4 are hybrid models offering both instant and extended thinking modes.

#### Temperature, Top-P and Max Tokens

```go
client, err := claude.New(ctx, apiKey,
    claude.WithTemperature(0.7),  // Optional: use either temperature OR top_p, not both
    // claude.WithTopP(0.9),      // Alternative to temperature
    claude.WithMaxTokens(8192),   // Optional (default: 8192)
)
```

**Note**: Claude Sonnet 4.5 does not allow both `temperature` and `top_p` to be specified simultaneously. Use one or the other.

### Environment Variables

- `ANTHROPIC_API_KEY` - Anthropic API key
- `GOLLEM_LOGGING_CLAUDE_PROMPT` - Enable prompt logging
- `GOLLEM_LOGGING_CLAUDE_RESPONSE` - Enable response logging

## Claude (Vertex AI)

### Basic Setup

```go
import (
    "context"
    "github.com/gollem-dev/gollem/llm/claude"
)

client, err := claude.NewWithVertex(ctx, "us-central1", "your-project-id")
```

### Configuration Options

#### Model Selection

```go
client, err := claude.NewWithVertex(ctx, region, projectID,
    claude.WithVertexModel("claude-sonnet-4@20250514"),
)
```

Available models on Vertex AI:
- `claude-opus-4-1@20250805` - Most powerful model (if available in your region)
- `claude-sonnet-4@20250514` - Latest Claude Sonnet model
- `claude-3-5-sonnet@20241022` - Previous generation Sonnet
- `claude-3-5-haiku@20241022` - Fast, cost-effective model

#### System Prompt

```go
client, err := claude.NewWithVertex(ctx, region, projectID,
    claude.WithVertexSystemPrompt("You are a helpful assistant."),
)
```

### Authentication

Uses Google Cloud credentials (same as Gemini):

```bash
# Option 1: Service account key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"

# Option 2: gcloud CLI
gcloud auth application-default login
```

### Benefits of Vertex AI Integration

- Unified Google Cloud billing and cost management
- Enterprise security with VPC, private endpoints, and audit logs
- Regional deployment for data residency requirements
- Vertex AI MLOps integration for monitoring and management

## OpenAI

### Basic Setup

```go
import (
    "context"
    "github.com/gollem-dev/gollem/llm/openai"
)

client, err := openai.New(ctx, "your-api-key")
```

### Configuration Options

#### Model Selection

```go
client, err := openai.New(ctx, apiKey,
    openai.WithModel("gpt-4-turbo-preview"),
)
```

Available models:
- `o3-pro` - Most powerful reasoning model with extended thinking
- `o3` - Advanced reasoning model for complex tasks
- `o4-mini` - Fast, cost-efficient reasoning model
- `gpt-4.1` - Latest GPT model with 1M token context (June 2024 cutoff)
- `gpt-4.1-mini` - Smaller version of GPT-4.1
- `gpt-4o` - Previous generation, still available
- `gpt-4o-mini` - Smaller, faster GPT-4o variant
- `gpt-3.5-turbo` - Legacy model, cost-effective

Note: GPT-4.5 is in research preview. o1 models are being phased out in favor of o3/o4 series.

#### Temperature and Other Parameters

```go
client, err := openai.New(ctx, apiKey,
    openai.WithTemperature(0.7),
    openai.WithMaxTokens(4096),  // Optional, omit for infinity (model's max)
    openai.WithTopP(0.9),
    openai.WithFrequencyPenalty(0.5),
    openai.WithPresencePenalty(0.5),
)
```

#### Organization and Base URL

```go
client, err := openai.New(ctx, apiKey,
    openai.WithOrganization("org-id"),
    openai.WithBaseURL("https://custom-endpoint.com"),
)
```

### Environment Variables

- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_ORGANIZATION` - Organization ID (optional)
- `GOLLEM_LOGGING_OPENAI_PROMPT` - Enable prompt logging
- `GOLLEM_LOGGING_OPENAI_RESPONSE` - Enable response logging

## PDF Input Support

gollem supports sending PDF documents to LLMs as input, enabling document analysis, extraction, and summarization.

### Creating PDF Input

```go
// From byte data
data, err := os.ReadFile("document.pdf")
if err != nil {
    return err
}
pdf, err := gollem.NewPDF(data)
if err != nil {
    return err
}

// From io.Reader
f, err := os.Open("document.pdf")
if err != nil {
    return err
}
defer f.Close()
pdf, err := gollem.NewPDFFromReader(f)
if err != nil {
    return err
}

// With custom max size
pdf, err := gollem.NewPDFFromReader(f, gollem.WithMaxPDFSize(64*1024*1024)) // 64MB
```

### Sending PDF to LLM

```go
result, err := session.Generate(ctx, []gollem.Input{
    pdf,
    gollem.Text("What are the key findings in this document?"),
})
if err != nil {
    return err
}
fmt.Println(result.Texts)
```

### Provider Compatibility

| Provider | PDF Support | Implementation |
|----------|------------|----------------|
| Claude (Anthropic) | Yes | Document block with base64-encoded data |
| Claude (Vertex AI) | Yes | Document block with base64-encoded data |
| Gemini | Yes | Inline data with `application/pdf` MIME type |
| OpenAI | No | OpenAI API does not accept PDF via the image_url field |

### Validation and Safety

- **Format validation**: PDF data must start with the `%PDF-` magic bytes
- **Size limit**: Default maximum is 32MB (`gollem.DefaultMaxPDFSize`), configurable via `gollem.WithMaxPDFSize()`
- **Memory protection**: `NewPDFFromReader` uses `io.LimitReader` internally to prevent reading unlimited data from untrusted sources

### History Round-Trip

PDF inputs are preserved during cross-provider history conversion. A PDF sent to Claude can be restored when converting history to Gemini format, and vice versa. OpenAI history uses `data:application/pdf;base64,...` data URLs for storage, though OpenAI's API does not support PDF input directly.

## Common Configuration Patterns

### Session Configuration

All LLM clients support common session options:

```go
session, err := client.NewSession(ctx,
    gollem.WithSessionHistory(history),
    gollem.WithSessionContentType(gollem.ContentTypeJSON),
    gollem.WithSessionTools(tool1, tool2),
    gollem.WithSessionSystemPrompt("You are a helpful assistant."),
)
```

### Per-Call Options

Override session defaults for a single `Generate` or `Stream` call:

```go
resp, err := session.Generate(ctx, inputs,
    gollem.WithTemperature(0.2),
    gollem.WithMaxTokens(256),
    gollem.WithGenerateResponseSchema(schema), // forces JSON output for this call
)
```

See [Per-Call Generate Options](schema.md#per-call-generate-options) for details.

### Embedding Generation

Providers that support embeddings (OpenAI and Gemini):

```go
embeddings, err := client.GenerateEmbedding(ctx, 
    768,           // dimension
    []string{      // texts to embed
        "Hello world",
        "Another text",
    },
)
```

### Error Handling

All providers return standardized errors that can be checked:

```go
resp, err := session.Generate(ctx, []gollem.Input{input})
if err != nil {
    // Check for specific error types
    // Handle token limit errors, rate limits, etc.
    return err
}
```

## Debugging and Monitoring

### Enable Logging

Use the `trace/logger` package to enable detailed logging for LLM interactions:

```go
import tracelogger "github.com/gollem-dev/gollem/trace/logger"

// Log LLM requests and responses
handler := tracelogger.New(
    tracelogger.WithEvents(tracelogger.LLMRequest, tracelogger.LLMResponse),
)

agent := gollem.New(client, gollem.WithTraceHandler(handler))
```

See [debugging.md](debugging.md) for full details on available events and configuration.

### Log Output Format

Logs are structured via `slog`:

```json
{
  "level": "INFO",
  "msg": "llm_call_end",
  "elapsed_ms": 1234,
  "texts": ["Generated response text"]
}
```