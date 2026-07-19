package gollem

import "context"

// Session is a session for the LLM. It maintains conversation state across
// multiple calls and can be used with the Agent (via Execute) or standalone
// for direct LLM interaction.
type Session interface {
	// Generate sends input to the LLM and returns the complete response.
	// Optional GenerateOption values override session-level defaults
	// (e.g. temperature, response schema) for this single call only.
	Generate(ctx context.Context, input []Input, opts ...GenerateOption) (*Response, error)

	// Stream sends input to the LLM and returns a channel that yields
	// response chunks as they arrive. Optional GenerateOption values
	// override session-level defaults for this single call only.
	// The channel is closed when the response is complete.
	Stream(ctx context.Context, input []Input, opts ...GenerateOption) (<-chan *Response, error)

	// Deprecated: Use Generate instead.
	GenerateContent(ctx context.Context, input ...Input) (*Response, error)
	// Deprecated: Use Stream instead.
	GenerateStream(ctx context.Context, input ...Input) (<-chan *Response, error)

	History() (*History, error)
	AppendHistory(*History) error
	CountToken(ctx context.Context, input ...Input) (int, error)
}

// SessionConfig is the configuration for the new session. This is required for only LLM client implementations.
type SessionConfig struct {
	history        *History
	contentType    ContentType
	systemPrompt   string
	tools          []Tool
	responseSchema *Parameter
	promptCache    bool

	// Middleware fields (ToolMiddleware excluded - managed at Agent layer)
	contentBlockMiddlewares  []ContentBlockMiddleware
	contentStreamMiddlewares []ContentStreamMiddleware
}

// History returns the history of the session.
func (c *SessionConfig) History() *History {
	return c.history
}

// SystemPrompt returns the system prompt of the session.
func (c *SessionConfig) SystemPrompt() string {
	return c.systemPrompt
}

// ContentType returns the content type of the session.
func (c *SessionConfig) ContentType() ContentType {
	return c.contentType
}

// Tools returns the tools of the session.
func (c *SessionConfig) Tools() []Tool {
	return c.tools
}

// ContentBlockMiddlewares returns the content block middlewares of the session.
func (c *SessionConfig) ContentBlockMiddlewares() []ContentBlockMiddleware {
	return c.contentBlockMiddlewares
}

// ContentStreamMiddlewares returns the content stream middlewares of the session.
func (c *SessionConfig) ContentStreamMiddlewares() []ContentStreamMiddleware {
	return c.contentStreamMiddlewares
}

// ResponseSchema returns the response schema of the session.
func (c *SessionConfig) ResponseSchema() *Parameter {
	return c.responseSchema
}

// PromptCache reports whether provider prompt caching is enabled for the session.
func (c *SessionConfig) PromptCache() bool {
	return c.promptCache
}

// NewSessionConfig creates a new session configuration. This is required for only LLM client implementations.
func NewSessionConfig(options ...SessionOption) SessionConfig {
	cfg := SessionConfig{}
	for _, option := range options {
		option(&cfg)
	}
	return cfg
}

// SessionOption is the option for the session configuration. This is required for only LLM client implementations.
type SessionOption func(cfg *SessionConfig)

// WithSessionHistory sets the history for the session.
// Usage:
// session, err := llmClient.NewSession(ctx, gollem.WithSessionHistory(history))
func WithSessionHistory(history *History) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.history = history
	}
}

// WithSessionContentType sets the content type for the session.
// Usage:
// session, err := llmClient.NewSession(ctx, gollem.WithSessionContentType(gollem.ContentTypeJSON))
func WithSessionContentType(contentType ContentType) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.contentType = contentType
	}
}

// WithSessionTools sets the tools for the session.
// Usage:
// session, err := llmClient.NewSession(ctx, gollem.WithSessionTools(tools))
func WithSessionTools(tools ...Tool) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.tools = append(cfg.tools, tools...)
	}
}

// WithSessionSystemPrompt sets the system prompt for the session.
// Usage:
// session, err := llmClient.NewSession(ctx, gollem.WithSessionSystemPrompt("You are a helpful assistant."))
func WithSessionSystemPrompt(systemPrompt string) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.systemPrompt = systemPrompt
	}
}

// WithSessionContentBlockMiddleware sets the content block middlewares for the session.
// Usage:
// session, err := llmClient.NewSession(ctx, gollem.WithSessionContentBlockMiddleware(middleware1, middleware2))
func WithSessionContentBlockMiddleware(middlewares ...ContentBlockMiddleware) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.contentBlockMiddlewares = append(cfg.contentBlockMiddlewares, middlewares...)
	}
}

// WithSessionContentStreamMiddleware sets the content stream middlewares for the session.
// Usage:
// session, err := llmClient.NewSession(ctx, gollem.WithSessionContentStreamMiddleware(middleware1, middleware2))
func WithSessionContentStreamMiddleware(middlewares ...ContentStreamMiddleware) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.contentStreamMiddlewares = append(cfg.contentStreamMiddlewares, middlewares...)
	}
}

// WithSessionResponseSchema sets the response schema for the session.
// The schema defines the structure of JSON output from the LLM.
// This option should be used with ContentTypeJSON.
//
// Usage:
//
//	schema := &gollem.Parameter{
//	    Title: "UserProfile",
//	    Description: "User profile information",
//	    Type: gollem.TypeObject,
//	    Properties: map[string]*gollem.Parameter{
//	        "name": {Type: gollem.TypeString, Description: "User name", Required: true},
//	        "age": {Type: gollem.TypeInteger, Description: "User age"},
//	    },
//	}
//	session, err := client.NewSession(ctx,
//	    gollem.WithSessionContentType(gollem.ContentTypeJSON),
//	    gollem.WithSessionResponseSchema(schema))
func WithSessionResponseSchema(schema *Parameter) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.responseSchema = schema
	}
}

// WithSessionPromptCache enables provider prompt caching for the session.
// Currently only the Claude provider acts on it: when enabled it injects
// ephemeral cache_control breakpoints on the stable prefix (system prompt and
// tools) and on the growing conversation tail, so repeated prefixes are served
// from Claude's prompt cache. OpenAI and Gemini cache automatically regardless
// of this flag; it does not change their requests. Default: disabled.
//
// Usage:
// session, err := client.NewSession(ctx, gollem.WithSessionPromptCache(true))
func WithSessionPromptCache(enabled bool) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.promptCache = enabled
	}
}

// ContentType represents the type of content to be generated by the LLM.
type ContentType string

const (
	// ContentTypeText represents plain text content.
	ContentTypeText ContentType = "text"
	// ContentTypeJSON represents JSON content.
	ContentTypeJSON ContentType = "json"
)
