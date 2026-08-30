package claude

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
)

const (
	// Default Claude models available via Vertex AI using Anthropic SDK
	DefaultVertexClaudeModel = "claude-sonnet-4@20250514"
)

// VertexClient is a client for Claude models via Vertex AI using official Anthropic SDK.
type VertexClient struct {
	// client is the underlying Anthropic client configured for Vertex AI.
	client *anthropic.Client

	// defaultModel is the model to use for chat completions.
	defaultModel string

	// embeddingModel is the model to use for embeddings.
	embeddingModel string

	// generation parameters
	params generationParameters

	// systemPrompt is the system prompt to use for chat completions.
	systemPrompt string
}

// VertexOption is a function that configures a VertexClient.
type VertexOption func(*VertexClient)

// WithVertexModel sets the default model to use for chat completions.
func WithVertexModel(modelName string) VertexOption {
	return func(c *VertexClient) {
		c.defaultModel = modelName
	}
}

// WithVertexEmbeddingModel sets the embedding model to use for embeddings.
func WithVertexEmbeddingModel(modelName string) VertexOption {
	return func(c *VertexClient) {
		c.embeddingModel = modelName
	}
}

// WithVertexTemperature sets the temperature parameter for text generation.
func WithVertexTemperature(temp float64) VertexOption {
	return func(c *VertexClient) {
		c.params.Temperature = temp
	}
}

// WithVertexTopP sets the top_p parameter for text generation.
func WithVertexTopP(topP float64) VertexOption {
	return func(c *VertexClient) {
		c.params.TopP = topP
	}
}

// WithVertexMaxTokens sets the maximum number of tokens to generate.
// When not set, the model's documented maximum output tokens is used.
// A value the API does not accept, such as zero or one above the model's
// limit, is sent as given and rejected by the API.
func WithVertexMaxTokens(maxTokens int64) VertexOption {
	return func(c *VertexClient) {
		c.params.MaxTokens = maxTokens
		c.params.maxTokensSet = true
	}
}

// WithVertexSystemPrompt sets the system prompt for the client.
func WithVertexSystemPrompt(prompt string) VertexOption {
	return func(c *VertexClient) {
		c.systemPrompt = prompt
	}
}

// newConfiguredVertexClient builds a VertexClient from the defaults and the
// given options, stopping short of the parts that need GCP credentials. It is
// split out of NewWithVertex so that tests can exercise the real defaults and
// option handling without reaching Vertex AI.
func newConfiguredVertexClient(options ...VertexOption) *VertexClient {
	client := &VertexClient{
		defaultModel:   DefaultVertexClaudeModel,
		embeddingModel: "text-embedding-004",
		params: generationParameters{
			Temperature: -1.0, // -1 indicates not set (0.0 is valid)
			TopP:        -1.0, // -1 indicates not set (0.0 is valid)
		},
	}

	for _, opt := range options {
		opt(client)
	}

	// The Messages API requires max_tokens, so it cannot be omitted. Fill in
	// the model's documented ceiling only when the caller did not choose one.
	if !client.params.maxTokensSet {
		client.params.MaxTokens = resolveMaxOutputTokens(client.defaultModel)
	}

	return client
}

// NewWithVertex creates a new client for Claude models via Vertex AI using Anthropic's official SDK.
// This is the recommended approach as it uses Anthropic's native Vertex AI integration.
func NewWithVertex(ctx context.Context, region, projectID string, options ...VertexOption) (*VertexClient, error) {
	if region == "" {
		return nil, goerr.New("region is required")
	}
	if projectID == "" {
		return nil, goerr.New("projectID is required")
	}

	client := newConfiguredVertexClient(options...)

	// Create Anthropic client with Vertex AI integration
	anthropicClient := anthropic.NewClient(
		option.WithAPIKey("dummy"), // Not used for Vertex AI
		vertex.WithGoogleAuth(ctx, region, projectID),
	)

	client.client = &anthropicClient

	return client, nil
}

// VertexAnthropicSession is a session for Claude via Vertex AI using Anthropic SDK.
type VertexAnthropicSession struct {
	client       *anthropic.Client
	defaultModel string
	params       generationParameters
	cfg          gollem.SessionConfig
	messages     []anthropic.MessageParam
}

// Model returns the model name this client generates through. It is the name
// the client was configured with, so a caller can key its own tables by the
// same string it passed to WithVertexModel.
func (c *VertexClient) Model() string { return c.defaultModel }

// NewSession creates a new session for Claude via Vertex AI using Anthropic SDK.
func (c *VertexClient) NewSession(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
	cfg := gollem.NewSessionConfig(options...)

	var messages []anthropic.MessageParam
	if cfg.History() != nil {
		history, err := ToMessages(cfg.History())
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert history to anthropic.MessageParam")
		}
		messages = append(messages, history...)
	}

	session := &VertexAnthropicSession{
		client:       c.client,
		defaultModel: c.defaultModel,
		params:       c.params,
		cfg:          cfg,
		messages:     messages,
	}

	return session, nil
}

// History returns the conversation history
func (s *VertexAnthropicSession) History() (*gollem.History, error) {
	return NewHistory(s.messages)
}

func (s *VertexAnthropicSession) AppendHistory(h *gollem.History) error {
	if h == nil {
		return nil
	}
	messages, err := ToMessages(h)
	if err != nil {
		return goerr.Wrap(err, "failed to convert history to Claude format")
	}
	s.messages = append(s.messages, messages...)
	return nil
}

// convertInputs converts gollem.Input to Claude messages and tool results
func (s *VertexAnthropicSession) convertInputs(ctx context.Context, input ...gollem.Input) ([]anthropic.MessageParam, []anthropic.ContentBlockParamUnion, error) {
	return convertGollemInputsToClaude(ctx, input...)
}

// Generate processes the input and generates a response with optional per-call overrides.
func (s *VertexAnthropicSession) Generate(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
	messages, _, err := s.convertInputs(ctx, input...)
	if err != nil {
		return nil, err
	}

	// Create a copy of messages for the API call, but don't update session history yet
	apiMessages := append([]anthropic.MessageParam{}, s.messages...)
	apiMessages = append(apiMessages, messages...)

	// Convert gollem tools to anthropic tools
	var tools []anthropic.ToolUnionParam
	if len(s.cfg.Tools()) > 0 {
		tools = make([]anthropic.ToolUnionParam, len(s.cfg.Tools()))
		for i, tool := range s.cfg.Tools() {
			tools[i] = convertTool(tool)
		}
	}

	// Build system prompt
	systemPrompt, err := createSystemPrompt(ctx, s.cfg)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to create system prompt")
	}

	// Start LLM call trace span
	var traceData *trace.LLMCallData
	var llmErr error
	if h := trace.HandlerFrom(ctx); h != nil {
		ctx = h.StartLLMCall(ctx)
		defer func() { h.EndLLMCall(ctx, traceData, llmErr) }()
	}

	// Build request
	msgParams := anthropic.MessageNewParams{
		Model:     anthropic.Model(s.defaultModel),
		MaxTokens: s.params.MaxTokens,
		Messages:  apiMessages,
	}
	if err := setTemperatureAndTopP(&msgParams, s.params.Temperature, s.params.TopP); err != nil {
		return nil, goerr.Wrap(err, "failed to set generation parameters")
	}
	if len(tools) > 0 {
		msgParams.Tools = tools
	}
	if len(systemPrompt) > 0 {
		msgParams.System = systemPrompt
	}

	// Apply per-call overrides
	if err := applyPerCallOverrides(&msgParams, opts...); err != nil {
		return nil, err
	}

	// Inject prompt-cache breakpoints on the stable prefix and tail
	if s.cfg.PromptCache() {
		applyPromptCacheBreakpoints(&msgParams)
	}

	resp, err := s.client.Messages.New(ctx, msgParams, option.WithRequestTimeout(defaultNonStreamingTimeout))
	if err != nil {
		llmErr = err
		opts := tokenLimitErrorOptions(err)
		return nil, goerr.Wrap(err, "failed to create message via Claude Vertex", opts...)
	}
	if err != nil {
		llmErr = err
		return nil, err
	}

	// Set trace data for defer.
	// Record only messages added in this turn; previous turns are already
	// captured in earlier trace spans.
	traceData = buildClaudeTraceData(resp, s.defaultModel, s.cfg.SystemPrompt(), messages)

	// Only update session history after successful API call
	s.messages = append(s.messages, messages...)

	// Only add response to history if it has content
	respParam := resp.ToParam()
	if len(respParam.Content) > 0 {
		s.messages = append(s.messages, respParam)
	}

	// Use JSON content type if per-call schema is set
	effectiveCT, hasSchema := effectiveContentType(s.cfg.ContentType(), s.cfg.ResponseSchema(), opts...)
	return processResponseWithContentType(ctx, resp, effectiveCT, hasSchema), nil
}

// Stream processes the input and generates a response stream with optional per-call overrides.
func (s *VertexAnthropicSession) Stream(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (<-chan *gollem.Response, error) {
	messages, _, err := s.convertInputs(ctx, input...)
	if err != nil {
		return nil, err
	}

	s.messages = append(s.messages, messages...)

	// Convert gollem tools to anthropic tools
	var tools []anthropic.ToolUnionParam
	if len(s.cfg.Tools()) > 0 {
		tools = make([]anthropic.ToolUnionParam, len(s.cfg.Tools()))
		for i, tool := range s.cfg.Tools() {
			tools[i] = convertTool(tool)
		}
	}

	// Build a temporary request to compute the system prompt override via applyPerCallOverrides
	var systemPromptOverride []anthropic.TextBlockParam
	genCfg := gollem.NewGenerateConfig(opts...)
	if genCfg.ResponseSchema() != nil {
		tmpRequest := anthropic.MessageNewParams{}
		systemPrompt, err := createSystemPrompt(ctx, s.cfg)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to create system prompt")
		}
		if len(systemPrompt) > 0 {
			tmpRequest.System = systemPrompt
		}
		if err := applyPerCallOverrides(&tmpRequest, opts...); err != nil {
			return nil, err
		}
		systemPromptOverride = tmpRequest.System
	}

	// Apply per-call overrides to a copy of params for Temperature/TopP/MaxTokens
	params := s.params
	if t := genCfg.Temperature(); t != nil {
		params.Temperature = *t
	}
	if p := genCfg.TopP(); p != nil {
		params.TopP = *p
	}
	if m := genCfg.MaxTokens(); m != nil {
		params.MaxTokens = int64(*m)
	}

	// Start LLM call trace span
	traceHandler := trace.HandlerFrom(ctx)
	if traceHandler != nil {
		ctx = traceHandler.StartLLMCall(ctx)
	}

	ch, err := generateClaudeStream(
		ctx,
		s.client,
		s.messages,
		s.defaultModel,
		params,
		tools,
		s.cfg,
		&s.messages,
		systemPromptOverride,
	)
	if err != nil {
		if traceHandler != nil {
			traceHandler.EndLLMCall(ctx, nil, err)
		}
		return nil, err
	}

	if traceHandler == nil {
		return ch, nil
	}

	// Wrap channel to capture trace data on stream completion
	wrappedCh := make(chan *gollem.Response)
	go func() {
		defer close(wrappedCh)

		var streamTraceData *trace.LLMCallData
		var streamErr error
		defer func() { traceHandler.EndLLMCall(ctx, streamTraceData, streamErr) }()

		var allTexts []string
		var allFunctionCalls []*trace.FunctionCall
		var lastInputTokens, lastOutputTokens int
		var lastCacheCreation, lastCacheRead int

		for resp := range ch {
			if resp.Error != nil && streamErr == nil {
				streamErr = resp.Error
			}
			allTexts = append(allTexts, resp.Texts...)
			for _, fc := range resp.FunctionCalls {
				allFunctionCalls = append(allFunctionCalls, &trace.FunctionCall{
					ID:        fc.ID,
					Name:      fc.Name,
					Arguments: fc.Arguments,
				})
			}
			if resp.InputToken > 0 {
				lastInputTokens = resp.InputToken
			}
			if resp.OutputToken > 0 {
				lastOutputTokens = resp.OutputToken
			}
			if resp.CacheCreationInputToken > 0 {
				lastCacheCreation = resp.CacheCreationInputToken
			}
			if resp.CacheReadInputToken > 0 {
				lastCacheRead = resp.CacheReadInputToken
			}
			wrappedCh <- resp
		}

		streamTraceData = &trace.LLMCallData{
			InputTokens:              lastInputTokens,
			OutputTokens:             lastOutputTokens,
			CacheCreationInputTokens: lastCacheCreation,
			CacheReadInputTokens:     lastCacheRead,
			Model:                    s.defaultModel,
			Request: &trace.LLMRequest{
				SystemPrompt: s.cfg.SystemPrompt(),
				// Record only messages added in this turn; previous turns are
				// already captured in earlier trace spans.
				Messages: claudeMessagesToTraceMessages(messages),
			},
			Response: &trace.LLMResponse{
				Texts:         allTexts,
				FunctionCalls: allFunctionCalls,
			},
		}
	}()

	return wrappedCh, nil
}

// Deprecated: GenerateContent is deprecated. Use Generate instead.
func (s *VertexAnthropicSession) GenerateContent(ctx context.Context, input ...gollem.Input) (*gollem.Response, error) {
	return s.Generate(ctx, input)
}

// Deprecated: GenerateStream is deprecated. Use Stream instead.
func (s *VertexAnthropicSession) GenerateStream(ctx context.Context, input ...gollem.Input) (<-chan *gollem.Response, error) {
	return s.Stream(ctx, input)
}

// CountToken calculates the total number of tokens for the given inputs,
// including system prompt, history messages, and new inputs.
// This uses Anthropic's Messages Count Tokens API via Vertex AI.
func (s *VertexAnthropicSession) CountToken(ctx context.Context, input ...gollem.Input) (int, error) {
	// Convert inputs to Claude messages
	messages, _, err := s.convertInputs(ctx, input...)
	if err != nil {
		return 0, goerr.Wrap(err, "failed to convert inputs for token counting")
	}

	// Create a copy of messages to avoid race conditions
	// This ensures thread safety when reading session state
	messagesCopy := make([]anthropic.MessageParam, len(s.messages))
	copy(messagesCopy, s.messages)

	// Convert tools from gollem.Tool to anthropic.ToolUnionParam
	var tools []anthropic.ToolUnionParam
	if len(s.cfg.Tools()) > 0 {
		tools = make([]anthropic.ToolUnionParam, 0, len(s.cfg.Tools()))
		for _, tool := range s.cfg.Tools() {
			tools = append(tools, convertTool(tool))
		}
	}

	// Use the shared helper function with a wrapper for the Vertex client
	apiClient := &realAPIClient{client: s.client}
	return countTokensWithParams(
		ctx,
		s.defaultModel,
		messagesCopy,
		messages,
		s.cfg.SystemPrompt(),
		tools,
		apiClient,
	)
}

// GenerateEmbedding generates embeddings for the given input texts.
func (c *VertexClient) GenerateEmbedding(ctx context.Context, dimension int, input []string) ([][]float64, error) {
	return nil, goerr.New("embedding generation not supported for Claude models via Vertex AI")
}
