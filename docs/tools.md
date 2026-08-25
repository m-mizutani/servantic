# Tools in gollem

Tools are your own custom built-in functions that LLMs can use to perform specific actions in your application. This guide explains how to create and use tools with gollem.

## Creating a Tool

To create a tool, you need to implement the `Tool` interface:

```go
type Tool interface {
    Spec() ToolSpec
    Run(ctx context.Context, args map[string]any) (map[string]any, error)
}
```

Here's an example of a simple tool:

```go
type HelloTool struct{}

func (t *HelloTool) Spec() gollem.ToolSpec {
    return gollem.ToolSpec{
        Name:        "hello",
        Description: "Returns a greeting",
        Parameters: map[string]*gollem.Parameter{
            "name": {
                Type:        gollem.TypeString,
                Description: "Name of the person to greet",
            },
        },
        Required: []string{"name"},
    }
}

func (t *HelloTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
    return map[string]any{
        "message": fmt.Sprintf("Hello, %s!", args["name"]),
    }, nil
}
```

## Type-safe Tools with `NewTool`

Implementing `Tool` by hand means writing the `Spec()` parameter map and then
asserting each argument's type inside `Run` (`args["name"].(string)`). Those two
places can drift apart, and a wrong assertion panics at runtime.

`NewTool[In, Out]` removes both problems. The input schema is inferred from the
`In` type (using the same struct tags as [`ToSchema`](schema.md#supported-struct-tags)),
and your handler receives a decoded `In` value instead of a raw map:

```go
type greetArgs struct {
    Name string `json:"name" description:"Name of the person to greet" required:"true"`
}

type greetResult struct {
    Message string `json:"message"`
}

tool, err := gollem.NewTool("hello", "Returns a greeting",
    func(ctx context.Context, in greetArgs) (greetResult, error) {
        return greetResult{Message: fmt.Sprintf("Hello, %s!", in.Name)}, nil
    })
if err != nil {
    return err
}
```

- `In` must be a **struct**. The handler argument is fully typed — no assertions,
  no panics. (A map is rejected: it would produce a property-less schema that tells
  the LLM nothing about the arguments, while giving none of the type safety.)
- `Out` must encode to a JSON object (a struct, or `map[string]any`). A
  `map[string]any` result is passed through unchanged, so existing tools can adopt
  `NewTool` incrementally by keeping their map output.
- **By default**, argument validation against the inferred schema runs automatically
  before the handler (see [Using Tools](#using-tools)), and the typed decode is the
  safety net behind it. If you disable it with `WithDisableArgsValidation`, only the
  typed decode remains: it enforces field *types*, but not `required`/`enum`/`min`/`max`,
  so a missing field arrives as its zero value.

`NewTool` returns an error when `In`/`Out` cannot form a valid schema (for example
a malformed struct tag, or a scalar type). For static registration, use
`MustNewTool`, which panics instead — convenient inside a slice literal:

```go
agent := gollem.New(client, gollem.WithTools(
    gollem.MustNewTool("hello", "Returns a greeting", greet),
    gollem.MustNewTool("add", "Adds two numbers", add),
))
```

### Validating Tool Types

To validate a tool's `In`/`Out` types without constructing the tool, use
`ToolSchema[In, Out]()` (returns an error) or `MustToolSchema[In, Out]()` (panics).
Both reuse the same inference as `NewTool`, so they agree with what it produces.

The primary use is a **startup assertion**: a malformed `In`/`Out` type (a bad
struct tag, a scalar type, etc.) is a programming error, and you want it to surface
when the program starts rather than on the first LLM call. Declare a package-level
assertion so the binary fails to start if the types ever become invalid:

```go
// Fail fast at package initialization if greetArgs/greetResult ever stop
// forming a valid tool schema (e.g. a struct tag is broken in a refactor).
var _ = gollem.MustToolSchema[greetArgs, greetResult]()
```

In a test, prefer the error-returning `ToolSchema` so the failure is reported as a
test failure rather than a panic:

```go
func TestGreetToolSchema(t *testing.T) {
    if _, err := gollem.ToolSchema[greetArgs, greetResult](); err != nil {
        t.Fatalf("invalid tool schema: %v", err)
    }
}
```

> [!NOTE]
> `ToolSchema` (tool input/output check) is distinct from `ToSchema` (single
> struct → schema). See [schema.md](schema.md#creating-schemas-from-go-structs).

## Tool Specification

The `ToolSpec` defines the tool's interface:

- `Name`: Unique identifier for the tool
- `Description`: Human-readable description of what the tool does
- `Parameters`: Map of parameter names to their specifications
- `Required`: List of required parameter names (For Object type)

Each parameter specification includes:
- `Type`: Parameter type (string, number, boolean, etc.)
- `Description`: Human-readable description
- `Title`: Optional user-friendly name for the parameter
- `Required`: Optional boolean indicating if the parameter is required
- `RequiredFields`: List of required field names when Type is Object
- `Enum`: Optional list of allowed values
- `Properties`: Map of properties when Type is Object
- `Items`: Specification for array items when Type is Array
- `Minimum`/`Maximum`: Number constraints
- `MinLength`/`MaxLength`: String length constraints
- `Pattern`: Regular expression pattern for string validation
- `MinItems`/`MaxItems`: Array size constraints
- `Default`: Default value for the parameter

> [!CAUTION]
> Note that not all parameters are supported by every LLM, as parameter support varies between different LLM providers.

### Numbers in `args` and in tool results

Numbers in the `args` map are `float64`, the type `encoding/json` produces. The one
exception is an integer a `float64` cannot represent exactly — one wider than 53 bits, such
as a large account ID or a nanosecond timestamp. Those arrive as `json.Number` so the value
the model sent is not rounded on the way in, and so the value a tool returns is not rounded
on the way back out.

Every number that a `float64` represents exactly is still a `float64`, so `args["n"].(float64)`
keeps working wherever it works today. A tool that accepts identifiers of that size should
handle both:

```go
func toInt64(v any) (int64, bool) {
    switch n := v.(type) {
    case float64:
        return int64(n), true
    case json.Number:
        i, err := n.Int64()
        return i, err == nil
    }
    return 0, false
}
```

> [!NOTE]
> This holds for OpenAI and Claude. It does not hold for Gemini: `google.golang.org/genai`
> rebuilds the whole request through `json.Marshal` followed by a plain `json.Unmarshal`
> into `map[string]any` before sending it, so an integer wider than 53 bits is rounded
> inside the SDK regardless of what gollem passes it. Pass such an identifier as a string
> when the tool is used with Gemini.

## Using Tools

To use tools with your agent:

```go
agent := gollem.New(client,
    gollem.WithTools(&HelloTool{}),
)

// Execute with automatic session management
err := agent.Execute(ctx, "Say hello to Alice")
if err != nil {
    panic(err)
}
```

You can add multiple tools:

```go
agent := gollem.New(client,
    gollem.WithTools(&HelloTool{}, &CalculatorTool{}, &WeatherTool{}),
    gollem.WithContentBlockMiddleware(func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
        return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
            resp, err := next(ctx, req)
            if err == nil && len(resp.Texts) > 0 {
                for _, text := range resp.Texts {
                    fmt.Printf("🤖 %s\n", text)
                }
            }
            return resp, err
        }
    }),
    gollem.WithToolMiddleware(func(next gollem.ToolHandler) gollem.ToolHandler {
        return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
            fmt.Printf("⚡ Executing: %s\n", req.Tool.Name)
            return next(ctx, req)
        }
    }),
)

// Execute multiple interactions with tools
err := agent.Execute(ctx, "Say hello to Bob and then calculate 15 + 27")
if err != nil {
    panic(err)
}
```

## Tool Monitoring with Middleware

gollem provides comprehensive middleware for monitoring tool execution:

```go
agent := gollem.New(client,
    gollem.WithTools(&HelloTool{}, &CalculatorTool{}),

    // Monitor and control tool execution
    gollem.WithToolMiddleware(func(next gollem.ToolHandler) gollem.ToolHandler {
        return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
            fmt.Printf("⚡ Executing tool: %s with args: %v\n", req.Tool.Name, req.Tool.Arguments)

            // Implement access control
            if !isToolAllowed(req.Tool.Name) {
                return &gollem.ToolExecResponse{
                    Error: fmt.Errorf("tool %s not allowed", req.Tool.Name),
                }, nil
            }

            // Execute tool
            resp, err := next(ctx, req)

            // Handle response
            if resp.Error != nil {
                fmt.Printf("❌ Tool %s failed: %v\n", req.Tool.Name, resp.Error)

                // Decide whether to continue or abort
                if isCriticalTool(req.Tool.Name) {
                    return resp, err // Abort execution
                }
            } else {
                fmt.Printf("✅ Tool %s completed: %v\n", req.Tool.Name, resp.Result)
            }

            return resp, err
        }
    }),
)
```


## SubAgents

SubAgents allow a parent agent to delegate tasks to specialized child agents. SubAgents implement the `Tool` interface, so they can be invoked by the LLM just like regular tools.

### Basic Usage (Default Mode)

In default mode, the subagent accepts a single `query` parameter:

```go
// Create a SubAgent with a factory function
// The factory is called each time the SubAgent is invoked,
// ensuring independent session state for each call
// The factory returns (*Agent, error) to handle creation failures
reviewSubagent := gollem.NewSubAgent(
    "code_reviewer",
    "Reviews code for best practices and potential issues",
    func() (*gollem.Agent, error) {
        return gollem.New(reviewClient,
            gollem.WithSystemPrompt("You are an expert code reviewer."),
        ), nil
    },
)

// Add to parent agent
parentAgent := gollem.New(client,
    gollem.WithSubAgents(reviewSubagent),
)

// LLM can now call: code_reviewer(query: "Review this function for security issues...")
```

### Template Mode

For more structured inputs, use `PromptTemplate` to define custom parameters:

```go
// Create a prompt template with custom parameters
template, err := gollem.NewPromptTemplate(
    `Analyze the following code focusing on {{.focus}}:

{{.code}}

Provide detailed feedback.`,
    map[string]*gollem.Parameter{
        "code":  {Type: gollem.TypeString, Description: "Code to analyze", Required: true},
        "focus": {Type: gollem.TypeString, Description: "Focus area (security, performance, etc.)", Required: true},
    },
)
if err != nil {
    panic(err)
}

// Create subagent with custom template using a factory function
analyzer := gollem.NewSubAgent(
    "code_analyzer",
    "Analyzes code with specified focus area",
    func() (*gollem.Agent, error) {
        return gollem.New(analyzerClient,
            gollem.WithSystemPrompt("You are a code analysis expert."),
        ), nil
    },
    gollem.WithPromptTemplate(template),
)

// LLM can now call: code_analyzer(code: "func main()...", focus: "security")
```

### Testing Templates

You can test your templates independently using the `Render` method:

```go
// Create template
template, err := gollem.NewPromptTemplate(
    "Hello, {{.name}}! Your task is: {{.task}}",
    map[string]*gollem.Parameter{
        "name": {Type: gollem.TypeString, Description: "User name", Required: true},
        "task": {Type: gollem.TypeString, Description: "Task description", Required: true},
    },
)
if err != nil {
    panic(err)
}

// Test rendering
result, err := template.Render(map[string]any{
    "name": "Alice",
    "task": "Review the code",
})
// result: "Hello, Alice! Your task is: Review the code"

// Inspect parameters
params := template.Parameters()
fmt.Printf("Required: %v\n", params["name"].Required) // true
```

### Default Template

Use `DefaultPromptTemplate()` to get the standard query-based template:

```go
template := gollem.DefaultPromptTemplate()
// Equivalent to: NewPromptTemplate("{{.query}}", map[string]*Parameter{"query": {...}})
```

### Nested SubAgents

SubAgents can have their own subagents for complex hierarchical workflows:

```go
// Level 3: Specialized subagents
securitySubagent := gollem.NewSubAgent(
    "security_check",
    "Security analysis",
    func() (*gollem.Agent, error) {
        return gollem.New(client, gollem.WithSystemPrompt("Security expert")), nil
    },
)

perfSubagent := gollem.NewSubAgent(
    "perf_check",
    "Performance analysis",
    func() (*gollem.Agent, error) {
        return gollem.New(client, gollem.WithSystemPrompt("Performance expert")), nil
    },
)

// Level 2: Code review agent with specialized subagents
reviewSubagent := gollem.NewSubAgent(
    "code_review",
    "Comprehensive code review",
    func() (*gollem.Agent, error) {
        return gollem.New(client,
            gollem.WithSystemPrompt("Code reviewer"),
            gollem.WithSubAgents(securitySubagent, perfSubagent),
        ), nil
    },
)

// Level 1: Main agent
mainAgent := gollem.New(client,
    gollem.WithSubAgents(reviewSubagent),
)
```

### SubAgent Middleware

SubAgent middleware allows you to inject context or modify arguments before template rendering. This is useful for adding runtime information (timestamps, user data, environment info) that the LLM doesn't know about.

```go
// Inject context information
subagent := gollem.NewSubAgent(
    "context_aware_analyzer",
    "Analyzes requests with user context",
    func() (*gollem.Agent, error) {
        return gollem.New(analyzerClient,
            gollem.WithSystemPrompt("You are an analyzer with user context awareness."),
        ), nil
    },
    gollem.WithPromptTemplate(prompt),
    gollem.WithSubAgentMiddleware(func(next gollem.SubAgentHandler) gollem.SubAgentHandler {
        return func(ctx context.Context, args map[string]any) (map[string]any, error) {
            // Add context that LLM doesn't provide
            user := getUserFromContext(ctx)
            args["current_time"] = time.Now().Format(time.RFC3339)
            args["user_name"] = user.Name
            args["user_role"] = user.Role
            return next(ctx, args)
        }
    }),
)
```

The injected context variables can be used in the template but are not exposed in the ToolSpec (the LLM doesn't see them as parameters):

```go
prompt, _ := gollem.NewPromptTemplate(
    `Current time: {{.current_time}}
User: {{.user_name}} ({{.user_role}})

Analyze the following request: {{.query}}`,
    map[string]*gollem.Parameter{
        // Only 'query' is visible to the LLM
        "query": {Type: gollem.TypeString, Description: "User query", Required: true},
    },
)
```

#### Chaining Multiple Middlewares

Multiple middlewares can be chained. They execute in the order they are added:

```go
subagent := gollem.NewSubAgent(
    "monitored_analyzer",
    "Analyzed with logging",
    func() (*gollem.Agent, error) {
        return gollem.New(client,
            gollem.WithSystemPrompt("You are an analyzer."),
        ), nil
    },
    gollem.WithPromptTemplate(prompt),
    // First middleware: logging
    gollem.WithSubAgentMiddleware(func(next gollem.SubAgentHandler) gollem.SubAgentHandler {
        return func(ctx context.Context, args map[string]any) (map[string]any, error) {
            log.Printf("SubAgent called with args: %v", args)
            result, err := next(ctx, args)
            log.Printf("SubAgent completed")
            return result, err
        }
    }),
    // Second middleware: context injection
    gollem.WithSubAgentMiddleware(func(next gollem.SubAgentHandler) gollem.SubAgentHandler {
        return func(ctx context.Context, args map[string]any) (map[string]any, error) {
            args["timestamp"] = time.Now().Unix()
            return next(ctx, args)
        }
    }),
)
```

#### Use Cases

- **Session information**: Inject user ID, session ID, timestamps
- **Environment context**: Current file path, working directory, project settings
- **External data**: Database lookups, API responses, cached data
- **Argument transformation**: Mask sensitive data, normalize inputs
- **Logging and monitoring**: Track invocations, measure latency

### Error Handling

SubAgent factory functions now return `(*Agent, error)` to handle creation failures gracefully. This allows proper error propagation and recovery strategies.

#### Factory Error Handling

```go
subagent := gollem.NewSubAgent(
    "configurable_agent",
    "Agent that requires valid configuration",
    func() (*gollem.Agent, error) {
        config, err := loadConfig()
        if err != nil {
            return nil, fmt.Errorf("failed to load config: %w", err)
        }

        client, err := createClient(config)
        if err != nil {
            return nil, fmt.Errorf("failed to create client: %w", err)
        }

        return gollem.New(client,
            gollem.WithSystemPrompt(config.SystemPrompt),
        ), nil
    },
)
```

#### Detecting Factory Errors with Middleware

Middlewares can detect and handle factory errors using `errors.Is()` with `gollem.ErrSubAgentFactory`:

```go
subagent := gollem.NewSubAgent(
    "resilient_agent",
    "Agent with error recovery",
    func() (*gollem.Agent, error) {
        return createAgent() // May fail
    },
    gollem.WithSubAgentMiddleware(func(next gollem.SubAgentHandler) gollem.SubAgentHandler {
        return func(ctx context.Context, args map[string]any) (map[string]any, error) {
            result, err := next(ctx, args)
            if err != nil && errors.Is(err, gollem.ErrSubAgentFactory) {
                // Factory error detected - provide fallback
                log.Error("Factory failed", "error", err)
                return map[string]any{
                    "response": "Service temporarily unavailable, using fallback",
                    "status":   "fallback",
                }, nil
            }
            return result, err
        }
    }),
)
```

#### Retry Pattern

Middlewares can implement retry logic for transient failures:

```go
subagent := gollem.NewSubAgent(
    "retry_agent",
    "Agent with automatic retry",
    func() (*gollem.Agent, error) {
        return createAgentWithRetry() // May fail transiently
    },
    gollem.WithSubAgentMiddleware(func(next gollem.SubAgentHandler) gollem.SubAgentHandler {
        return func(ctx context.Context, args map[string]any) (map[string]any, error) {
            result, err := next(ctx, args)
            if err != nil && errors.Is(err, gollem.ErrSubAgentFactory) {
                // Retry once on factory error
                log.Info("Retrying after factory error")
                return next(ctx, args)
            }
            return result, err
        }
    }),
)
```

#### Error Types

- **`gollem.ErrSubAgentFactory`**: Sentinel error wrapping all factory-related failures
  - Factory function returned an error
  - Factory function returned `nil` agent
  - Use `errors.Is(err, gollem.ErrSubAgentFactory)` to detect

## Next Steps

- Learn about [MCP server integration](mcp.md) for external tool integration
- Check out [practical examples](examples.md) of tool usage
- Review the [getting started guide](getting-started.md) for basic usage
- Understand [history management](history.md) for conversation context
- Explore the [complete documentation](README.md)
