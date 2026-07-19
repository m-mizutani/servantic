package gollem

import "context"

// ContentBlockMiddleware is a function that wraps a ContentBlockHandler to add behavior.
// Used for synchronous content generation.
type ContentBlockMiddleware func(next ContentBlockHandler) ContentBlockHandler

// ContentBlockHandler handles content generation requests synchronously.
type ContentBlockHandler func(ctx context.Context, req *ContentRequest) (*ContentResponse, error)

// ContentStreamMiddleware is a function that wraps a ContentStreamHandler to add behavior.
// Used for streaming content generation.
type ContentStreamMiddleware func(next ContentStreamHandler) ContentStreamHandler

// ContentStreamHandler handles content generation requests with streaming.
type ContentStreamHandler func(ctx context.Context, req *ContentRequest) (<-chan *ContentResponse, error)

// ContentRequest represents a request for content generation with modifiable history.
type ContentRequest struct {
	Inputs       []Input  // Current user inputs
	History      *History // Modifiable conversation history
	SystemPrompt string   // System prompt for this request
}

// ContentResponse represents a response from content generation.
type ContentResponse struct {
	Texts         []string        // Generated text content
	Thoughts      []string        // Thinking/reasoning content
	FunctionCalls []*FunctionCall // Function/tool call requests
	InputToken    int             // Number of input tokens used (total, incl. cache reads)
	OutputToken   int             // Number of output tokens used
	// CacheCreationInputToken is input tokens written to the prompt cache (Claude only).
	CacheCreationInputToken int
	// CacheReadInputToken is input tokens served from the prompt cache (cache hits).
	CacheReadInputToken int
	Error               error // Error if any occurred
}

// ToolMiddleware is a function that wraps a ToolHandler to add behavior.
// Used at Agent layer for tool execution interception.
type ToolMiddleware func(next ToolHandler) ToolHandler

// ToolHandler handles tool execution requests.
type ToolHandler func(ctx context.Context, req *ToolExecRequest) (*ToolExecResponse, error)

// ToolExecRequest represents a tool execution request.
type ToolExecRequest struct {
	Tool     *FunctionCall // Tool call details
	ToolSpec *ToolSpec     // Tool specification
}

// ToolExecResponse represents a tool execution response.
type ToolExecResponse struct {
	Result   map[string]any // Execution result
	Error    error          // Execution error if any
	Duration int64          // Execution duration in milliseconds
}

// BuildContentBlockChain builds a chain of ContentBlockMiddleware functions.
// The middlewares are applied in the order they are provided.
func BuildContentBlockChain(middlewares []ContentBlockMiddleware, handler ContentBlockHandler) ContentBlockHandler {
	// Apply middlewares in reverse order to maintain intuitive execution order
	for i := len(middlewares) - 1; i >= 0; i-- {
		handler = middlewares[i](handler)
	}
	return handler
}

// BuildContentStreamChain builds a chain of ContentStreamMiddleware functions.
// The middlewares are applied in the order they are provided.
func BuildContentStreamChain(middlewares []ContentStreamMiddleware, handler ContentStreamHandler) ContentStreamHandler {
	// Apply middlewares in reverse order to maintain intuitive execution order
	for i := len(middlewares) - 1; i >= 0; i-- {
		handler = middlewares[i](handler)
	}
	return handler
}

// buildToolChain builds a chain of ToolMiddleware functions.
// This is kept internal to the Agent layer.
func buildToolChain(middlewares []ToolMiddleware, handler ToolHandler) ToolHandler {
	// Apply middlewares in reverse order to maintain intuitive execution order
	for i := len(middlewares) - 1; i >= 0; i-- {
		handler = middlewares[i](handler)
	}
	return handler
}
