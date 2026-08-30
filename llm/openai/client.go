package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/internal/jsonutil"
	"github.com/gollem-dev/gollem/internal/schema"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
	"github.com/pkoukk/tiktoken-go"
	"github.com/sashabaranov/go-openai"
)

// generationParameters represents the parameters for text generation.
type generationParameters struct {
	// Temperature controls randomness in the output.
	// Higher values make the output more random, lower values make it more focused.
	Temperature float32

	// TopP controls diversity via nucleus sampling.
	// Higher values allow more diverse outputs.
	TopP float32

	// MaxTokens limits the number of tokens to generate.
	MaxTokens int

	// PresencePenalty increases the model's likelihood to talk about new topics.
	// Range: -2.0 to 2.0
	PresencePenalty float32

	// FrequencyPenalty decreases the model's likelihood to repeat the same line verbatim.
	// Range: -2.0 to 2.0
	FrequencyPenalty float32

	// ReasoningEffort tunes how much reasoning time the model spends ("minimal", "medium", "high").
	ReasoningEffort string

	// Verbosity controls the amount of output tokens generated ("low", "medium", "high").
	Verbosity string
}

// Client is a client for the OpenAI API.
// It provides methods to interact with OpenAI's OpenAI models.
type Client struct {
	// client is the underlying OpenAI client.
	client *openai.Client

	// defaultModel is the model to use for chat completions.
	// It can be overridden using WithModel option.
	defaultModel string

	// embeddingModel is the model to use for embeddings.
	// It can be overridden using WithEmbeddingModel option.
	embeddingModel string

	// baseURL is the custom base URL for the OpenAI API.
	// If empty, uses the default OpenAI API endpoints.
	baseURL string

	// generation parameters
	params generationParameters

	// systemPrompt is the system prompt to use for chat completions.
	systemPrompt string

	// contentType is the type of content to be generated.
	contentType gollem.ContentType
}

const (
	DefaultModel          = "gpt-5"
	DefaultEmbeddingModel = "text-embedding-3-small"
)

// Option is a function that configures a Client.
type Option func(*Client)

// WithModel sets the default model to use for chat completions.
// The model name should be a valid OpenAI model identifier.
// See default model in [DefaultModel].
func WithModel(modelName string) Option {
	return func(c *Client) {
		c.defaultModel = modelName
	}
}

// WithEmbeddingModel sets the embedding model to use for embeddings.
// The model name should be a valid OpenAI model identifier.
// See default embedding model in [DefaultEmbeddingModel].
// Model list is at https://platform.openai.com/docs/guides/embeddings#embedding-models
func WithEmbeddingModel(modelName string) Option {
	return func(c *Client) {
		c.embeddingModel = modelName
	}
}

// WithTemperature sets the temperature parameter for text generation.
// Higher values make the output more random, lower values make it more focused.
// Range: 0.0 to 1.0
// Default: 0.7
func WithTemperature(temp float32) Option {
	return func(c *Client) {
		c.params.Temperature = temp
	}
}

// WithTopP sets the top_p parameter for text generation.
// Controls diversity via nucleus sampling.
func WithTopP(topP float32) Option {
	return func(c *Client) {
		c.params.TopP = topP
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(maxTokens int) Option {
	return func(c *Client) {
		c.params.MaxTokens = maxTokens
	}
}

// WithPresencePenalty sets the presence penalty parameter.
// Increases the model's likelihood to talk about new topics.
func WithPresencePenalty(penalty float32) Option {
	return func(c *Client) {
		c.params.PresencePenalty = penalty
	}
}

// WithFrequencyPenalty sets the frequency penalty parameter.
// Decreases the model's likelihood to repeat the same line verbatim.
func WithFrequencyPenalty(penalty float32) Option {
	return func(c *Client) {
		c.params.FrequencyPenalty = penalty
	}
}

// WithReasoningEffort sets the reasoning_effort parameter for GPT-5 models.
// Supported values (as of 2025-10-04): "minimal", "medium", "high".
func WithReasoningEffort(effort string) Option {
	return func(c *Client) {
		c.params.ReasoningEffort = effort
	}
}

// WithVerbosity sets the verbosity parameter for GPT-5 models.
// Supported values (as of 2025-10-04): "low", "medium", "high".
func WithVerbosity(verbosity string) Option {
	return func(c *Client) {
		c.params.Verbosity = verbosity
	}
}

// WithSystemPrompt sets the system prompt to use for chat completions.
func WithSystemPrompt(prompt string) Option {
	return func(c *Client) {
		c.systemPrompt = prompt
	}
}

// WithContentType sets the content type for text generation.
// This determines the format of the generated content.
func WithContentType(contentType gollem.ContentType) Option {
	return func(c *Client) {
		c.contentType = contentType
	}
}

// WithBaseURL sets the custom base URL for the OpenAI API.
// Allows usage with compatible endpoints, proxies, or self-hosted instances.
// If empty, uses the default OpenAI API endpoints.
// Reference: Brain Memory c4705651-435d-4cca-95eb-d39d1ea69a9c
func WithBaseURL(url string) Option {
	return func(c *Client) {
		c.baseURL = url
	}
}

// New creates a new client for the OpenAI API.
// It requires an API key and can be configured with additional options.
func New(ctx context.Context, apiKey string, options ...Option) (*Client, error) {
	client := &Client{
		defaultModel:   DefaultModel,
		embeddingModel: DefaultEmbeddingModel,
		baseURL:        "", // Default empty, will be set by options
		params: generationParameters{
			ReasoningEffort: "minimal",
			Verbosity:       "low",
		},
		contentType: gollem.ContentTypeText,
	}

	for _, option := range options {
		option(client)
	}

	config := openai.DefaultConfig(apiKey)

	// Add BaseURL if specified
	if client.baseURL != "" {
		config.BaseURL = client.baseURL
	}

	openaiClient := openai.NewClientWithConfig(config)
	client.client = openaiClient

	return client, nil
}

// Session is a session for the OpenAI chat.
// It maintains the conversation state and handles message generation.
type Session struct {
	// apiClient is the API client interface for dependency injection.
	apiClient apiClient

	// defaultModel is the model to use for chat completions.
	defaultModel string

	// tools are the available tools for the session.
	tools []openai.Tool

	// currentHistory maintains the gollem.History for middleware access.
	historyMessages []openai.ChatCompletionMessage

	// generation parameters
	params generationParameters

	cfg gollem.SessionConfig

	// strictMode enables OpenAI's strict schema adherence (default: false)
	strictMode bool
}

// Model returns the model name this client generates through. It is the name
// the client was configured with, so a caller can key its own tables by the
// same string it passed to WithModel.
func (c *Client) Model() string { return c.defaultModel }

// NewSession creates a new session for the OpenAI API.
// It converts the provided tools to OpenAI's tool format and initializes a new chat session.
func (c *Client) NewSession(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
	cfg := gollem.NewSessionConfig(options...)

	// Convert gollem.Tool to openai.Tool
	openaiTools := make([]openai.Tool, len(cfg.Tools()))
	for i, tool := range cfg.Tools() {
		openaiTools[i] = convertTool(tool)
	}

	// Initialize history from config (convert to OpenAI native format)
	var historyMessages []openai.ChatCompletionMessage
	if cfg.History() != nil {
		var err error
		historyMessages, err = ToMessages(cfg.History())
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert history to OpenAI format")
		}
	}

	session := &Session{
		apiClient:       &realAPIClient{client: c.client},
		defaultModel:    c.defaultModel,
		tools:           openaiTools,
		params:          c.params,
		historyMessages: historyMessages,
		cfg:             cfg,
	}

	return session, nil
}

func (s *Session) History() (*gollem.History, error) {
	return NewHistory(s.historyMessages)
}

func (s *Session) AppendHistory(h *gollem.History) error {
	if h == nil {
		return nil
	}
	messages, err := ToMessages(h)
	if err != nil {
		return goerr.Wrap(err, "failed to convert history to OpenAI format")
	}
	s.historyMessages = append(s.historyMessages, messages...)
	return nil
}

// getMessages returns history messages (already in OpenAI format)
func (s *Session) getMessages() ([]openai.ChatCompletionMessage, error) {
	if len(s.historyMessages) == 0 {
		return []openai.ChatCompletionMessage{}, nil
	}
	messages := s.historyMessages

	return messages, nil
}

// updateHistoryWithResponse updates the current history with an assistant response
func (s *Session) updateHistoryWithResponse(assistantMessage openai.ChatCompletionMessage) error {
	// Get current messages and append the assistant response
	currentMessages, err := s.getMessages()
	if err != nil {
		return goerr.Wrap(err, "failed to get current messages")
	}
	allMessages := append(currentMessages, assistantMessage)

	// DEBUG: Debug logging can be enabled here for troubleshooting tool_call_id issues

	// Create new history from all messages
	s.historyMessages = allMessages
	return nil
}

// convertInputsToMessages converts gollem.Input to OpenAI messages without modifying session state.
// This is a pure function used for read-only operations like CountToken.
func (s *Session) convertInputsToMessages(input ...gollem.Input) ([]openai.ChatCompletionMessage, error) {
	var newMessages []openai.ChatCompletionMessage

	// Accumulate consecutive user content (Text/Image) into a single message
	var userContentParts []openai.ChatMessagePart

	for _, in := range input {
		switch v := in.(type) {
		case gollem.Text:
			userContentParts = append(userContentParts, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeText,
				Text: string(v),
			})

		case gollem.Image:
			// Create image URL in data format for OpenAI
			imageURL := fmt.Sprintf("data:%s;base64,%s", v.MimeType(), v.Base64())
			userContentParts = append(userContentParts, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL: imageURL,
				},
			})

		case gollem.PDF:
			// OpenAI SDK doesn't have native PDF support; use data URL in image_url field
			pdfURL := fmt.Sprintf("data:application/pdf;base64,%s", v.Base64())
			userContentParts = append(userContentParts, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL: pdfURL,
				},
			})

		case gollem.FunctionResponse:
			// If we have accumulated user content, create a message for it
			if len(userContentParts) > 0 {
				newMessages = append(newMessages, openai.ChatCompletionMessage{
					Role:         openai.ChatMessageRoleUser,
					MultiContent: userContentParts,
				})
				userContentParts = nil
			}
			data, err := json.Marshal(v.Data)
			if err != nil {
				return nil, goerr.Wrap(err, "failed to marshal function response")
			}
			response := string(data)
			if v.Error != nil {
				response = fmt.Sprintf(`Error message: %+v`, v.Error)
			}

			newMessages = append(newMessages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    response,
				ToolCallID: v.ID,
			})
		default:
			return nil, goerr.Wrap(gollem.ErrInvalidParameter, "invalid input")
		}
	}

	// Create final user message if there's any remaining user content
	if len(userContentParts) > 0 {
		newMessages = append(newMessages, openai.ChatCompletionMessage{
			Role:         openai.ChatMessageRoleUser,
			MultiContent: userContentParts,
		})
	}

	return newMessages, nil
}

// convertInputs converts gollem.Input to OpenAI messages, appends them to the
// session history, and returns the newly added messages so callers (e.g. trace)
// can record only the messages added in this turn.
func (s *Session) convertInputs(input ...gollem.Input) ([]openai.ChatCompletionMessage, error) {
	// Convert inputs to messages using the pure function
	newMessages, err := s.convertInputsToMessages(input...)
	if err != nil {
		return nil, err
	}

	// Update currentHistory with new messages
	if len(newMessages) > 0 {
		s.historyMessages = append(s.historyMessages, newMessages...)
	}

	return newMessages, nil
}

// createRequest creates a chat completion request with the current session state
func (s *Session) createRequest(stream bool) (openai.ChatCompletionRequest, error) {
	messages, err := s.getMessages()
	if err != nil {
		return openai.ChatCompletionRequest{}, goerr.Wrap(err, "failed to get messages for API call")
	}

	req := openai.ChatCompletionRequest{
		Model:               s.defaultModel,
		Messages:            messages,
		Tools:               s.tools,
		Temperature:         s.params.Temperature,
		TopP:                s.params.TopP,
		MaxCompletionTokens: s.params.MaxTokens,
		PresencePenalty:     s.params.PresencePenalty,
		FrequencyPenalty:    s.params.FrequencyPenalty,
		Stream:              stream,
	}

	if s.params.ReasoningEffort != "" {
		req.ReasoningEffort = s.params.ReasoningEffort
	}

	if s.params.Verbosity != "" {
		req.Verbosity = s.params.Verbosity
	}

	// Add content type and response schema to the request
	if s.cfg.ContentType() == gollem.ContentTypeJSON {
		if s.cfg.ResponseSchema() != nil {
			// Use structured outputs with schema
			schema, err := convertResponseSchemaToOpenAI(s.cfg.ResponseSchema(), s.strictMode)
			if err != nil {
				return openai.ChatCompletionRequest{}, goerr.Wrap(err, "failed to convert response schema")
			}
			req.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type:       openai.ChatCompletionResponseFormatTypeJSONSchema,
				JSONSchema: schema,
			}
		} else {
			// Use simple JSON object mode (existing behavior)
			req.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		}
	}

	return req, nil
}

// cachedPromptTokens returns the number of prompt tokens served from OpenAI's
// automatic prompt cache, or 0 when the API did not report cache details.
func cachedPromptTokens(u openai.Usage) int {
	if u.PromptTokensDetails != nil {
		return u.PromptTokensDetails.CachedTokens
	}
	return 0
}

// Generate processes the input and generates a response with optional per-call overrides.
// It handles both text messages and function responses.
func (s *Session) Generate(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
	// Build the content request for middleware
	// Create a copy of the current history to avoid middleware side effects
	var historyCopy *gollem.History
	var err error
	if len(s.historyMessages) > 0 {
		historyCopy, err = NewHistory(s.historyMessages)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to create history copy for middleware")
		}
	}

	contentReq := &gollem.ContentRequest{
		Inputs:       input,
		History:      historyCopy,
		SystemPrompt: s.cfg.SystemPrompt(),
	}

	// Create the base handler that performs the actual API call
	baseHandler := func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
		// Always update history from middleware (even if same address, content may have changed)
		if req.History != nil {
			var err error
			s.historyMessages, err = ToMessages(req.History)
			if err != nil {
				return nil, goerr.Wrap(err, "failed to convert history from middleware")
			}
		}

		// Convert inputs and perform the actual API call
		newMessages, err := s.convertInputs(req.Inputs...)
		if err != nil {
			return nil, err
		}

		openaiReq, err := s.createRequest(false)
		if err != nil {
			return nil, err
		}

		if err := s.applyPerCallOverrides(&openaiReq, opts...); err != nil {
			return nil, err
		}

		// Start LLM call trace span
		var openaiTraceData *trace.LLMCallData
		var llmErr error
		if h := trace.HandlerFrom(ctx); h != nil {
			ctx = h.StartLLMCall(ctx)
			defer func() { h.EndLLMCall(ctx, openaiTraceData, llmErr) }()
		}

		resp, err := s.apiClient.CreateChatCompletion(ctx, openaiReq)
		if err != nil {
			llmErr = err
			opts := tokenLimitErrorOptions(err)
			return nil, goerr.Wrap(err, "failed to create chat completion", opts...)
		}

		if len(resp.Choices) == 0 {
			openaiTraceData = &trace.LLMCallData{
				Model:    resp.Model,
				Response: &trace.LLMResponse{},
			}
			return &gollem.ContentResponse{
				Texts:         []string{},
				FunctionCalls: []*gollem.FunctionCall{},
				InputToken:    0,
				OutputToken:   0,
			}, nil
		}

		response := &gollem.Response{
			Texts:         make([]string, 0),
			Thoughts:      make([]string, 0),
			FunctionCalls: make([]*gollem.FunctionCall, 0),
			// OpenAI caches automatically; PromptTokens already includes cached
			// tokens, so InputToken stays total. Cached reads are reported for
			// observability; OpenAI does not report cache writes (creation stays 0).
			InputToken:          resp.Usage.PromptTokens,
			OutputToken:         resp.Usage.CompletionTokens,
			CacheReadInputToken: cachedPromptTokens(resp.Usage),
		}

		message := resp.Choices[0].Message
		if message.Content != "" {
			response.Texts = append(response.Texts, message.Content)
		}

		if message.ReasoningContent != "" {
			response.Thoughts = append(response.Thoughts, message.ReasoningContent)
		}

		if message.ToolCalls != nil {
			for _, toolCall := range message.ToolCalls {
				args, err := jsonutil.DecodeObject([]byte(toolCall.Function.Arguments))
				if err != nil {
					return nil, goerr.Wrap(err, "failed to unmarshal tool arguments")
				}

				response.FunctionCalls = append(response.FunctionCalls, &gollem.FunctionCall{
					ID:        toolCall.ID,
					Name:      toolCall.Function.Name,
					Arguments: args,
				})
			}

			// Create assistant message with all tool calls
			assistantMessage := openai.ChatCompletionMessage{
				Role:      openai.ChatMessageRoleAssistant,
				Content:   message.Content,
				ToolCalls: message.ToolCalls,
			}

			// Update history with assistant response
			if err := s.updateHistoryWithResponse(assistantMessage); err != nil {
				return nil, goerr.Wrap(err, "failed to update history with assistant response")
			}
		} else if message.Content != "" {
			// Create assistant message without tool calls
			assistantMessage := openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: message.Content,
			}

			// Update history with assistant response
			if err := s.updateHistoryWithResponse(assistantMessage); err != nil {
				return nil, goerr.Wrap(err, "failed to update history with assistant response")
			}
		}

		// Set trace data for defer.
		// Record only messages added in this turn; the full request history is
		// captured incrementally across previous trace spans.
		openaiTraceData = buildOpenAITraceData(resp, s.defaultModel, s.cfg.SystemPrompt(), newMessages)

		// History is already updated by updateHistoryWithResponse above

		return &gollem.ContentResponse{
			Texts:               response.Texts,
			Thoughts:            response.Thoughts,
			FunctionCalls:       response.FunctionCalls,
			InputToken:          response.InputToken,
			OutputToken:         response.OutputToken,
			CacheReadInputToken: response.CacheReadInputToken,
		}, nil
	}

	// Build middleware chain
	handler := gollem.ContentBlockHandler(baseHandler)
	for i := len(s.cfg.ContentBlockMiddlewares()) - 1; i >= 0; i-- {
		handler = s.cfg.ContentBlockMiddlewares()[i](handler)
	}

	// Execute middleware chain
	contentResp, err := handler(ctx, contentReq)
	if err != nil {
		return nil, err
	}

	// Update history after middleware execution (history was already updated in baseHandler)
	// Convert ContentResponse back to gollem.Response
	return &gollem.Response{
		Texts:               contentResp.Texts,
		Thoughts:            contentResp.Thoughts,
		FunctionCalls:       contentResp.FunctionCalls,
		InputToken:          contentResp.InputToken,
		OutputToken:         contentResp.OutputToken,
		CacheReadInputToken: contentResp.CacheReadInputToken,
	}, nil
}

// Stream processes the input and generates a response stream with optional per-call overrides.
// It handles both text messages and function responses, and returns a channel for streaming responses.
func (s *Session) Stream(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (<-chan *gollem.Response, error) {
	// Build the content request for middleware
	var historyCopy *gollem.History
	var err error
	if len(s.historyMessages) > 0 {
		historyCopy, err = NewHistory(s.historyMessages)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to create history copy for middleware")
		}
	}

	contentReq := &gollem.ContentRequest{
		Inputs:       input,
		History:      historyCopy,
		SystemPrompt: s.cfg.SystemPrompt(),
	}

	// Create the base handler that performs the actual API call
	baseHandler := func(ctx context.Context, req *gollem.ContentRequest) (<-chan *gollem.ContentResponse, error) {
		// Always update history from middleware (even if same address, content may have changed)
		if req.History != nil {
			var err error
			s.historyMessages, err = ToMessages(req.History)
			if err != nil {
				return nil, goerr.Wrap(err, "failed to convert history from middleware")
			}
		}

		// Convert inputs and perform the actual API call
		newMessages, err := s.convertInputs(req.Inputs...)
		if err != nil {
			return nil, err
		}

		openaiReq, err := s.createRequest(true)
		if err != nil {
			return nil, err
		}

		if err := s.applyPerCallOverrides(&openaiReq, opts...); err != nil {
			return nil, err
		}

		// Start LLM call trace span
		traceHandler := trace.HandlerFrom(ctx)
		if traceHandler != nil {
			ctx = traceHandler.StartLLMCall(ctx)
		}

		// Enable stream options to get usage data
		openaiReq.StreamOptions = &openai.StreamOptions{
			IncludeUsage: true,
		}
		stream, err := s.apiClient.CreateChatCompletionStream(ctx, openaiReq)
		if err != nil {
			if traceHandler != nil {
				traceHandler.EndLLMCall(ctx, nil, err)
			}
			opts := tokenLimitErrorOptions(err)
			return nil, goerr.Wrap(err, "failed to create chat completion stream", opts...)
		}

		responseChan := make(chan *gollem.ContentResponse)

		go func() {
			defer close(responseChan)
			defer func() { _ = stream.Close() }()

			var streamTraceData *trace.LLMCallData
			var streamErr error
			if traceHandler != nil {
				defer func() { traceHandler.EndLLMCall(ctx, streamTraceData, streamErr) }()
			}

			var textContent string
			var reasoningContent string
			var toolCalls []openai.ToolCall
			var totalInputTokens int
			var totalOutputTokens int
			var totalCacheRead int

			// Process streaming chunks
			for {
				select {
				case <-ctx.Done():
					responseChan <- &gollem.ContentResponse{
						Error: goerr.Wrap(ctx.Err(), "context cancelled during streaming"),
					}
					return
				default:
				}

				resp, err := stream.Recv()
				if err != nil {
					if err == io.EOF {
						break
					}
					opts := tokenLimitErrorOptions(err)
					responseChan <- &gollem.ContentResponse{
						Error: goerr.Wrap(err, "failed to receive chat completion stream", opts...),
					}
					return
				}

				// Handle token usage if available (comes in final chunk)
				if resp.Usage != nil {
					totalInputTokens = resp.Usage.PromptTokens
					totalOutputTokens = resp.Usage.CompletionTokens
					totalCacheRead = cachedPromptTokens(*resp.Usage)
				}

				if len(resp.Choices) == 0 {
					continue
				}

				choice := resp.Choices[0]
				delta := choice.Delta

				// Handle text content
				if delta.Content != "" {
					textContent += delta.Content
					responseChan <- &gollem.ContentResponse{
						Texts:               []string{delta.Content},
						InputToken:          totalInputTokens,
						OutputToken:         totalOutputTokens,
						CacheReadInputToken: totalCacheRead,
					}
				}

				// Handle reasoning content
				if delta.ReasoningContent != "" {
					reasoningContent += delta.ReasoningContent
					responseChan <- &gollem.ContentResponse{
						Thoughts:            []string{delta.ReasoningContent},
						InputToken:          totalInputTokens,
						OutputToken:         totalOutputTokens,
						CacheReadInputToken: totalCacheRead,
					}
				}

				// Handle tool calls - accumulate them
				if delta.ToolCalls != nil {
					for _, toolCall := range delta.ToolCalls {
						// Get the index, defaulting to 0 if nil
						index := 0
						if toolCall.Index != nil {
							index = *toolCall.Index
						}

						// Ensure we have enough space in the slice
						for len(toolCalls) <= index {
							toolCalls = append(toolCalls, openai.ToolCall{
								Function: openai.FunctionCall{},
							})
						}

						tc := &toolCalls[index]

						if toolCall.ID != "" {
							tc.ID = toolCall.ID
						}
						if toolCall.Type != "" {
							tc.Type = toolCall.Type
						}
						if toolCall.Function.Name != "" {
							tc.Function.Name = toolCall.Function.Name
						}
						if toolCall.Function.Arguments != "" {
							tc.Function.Arguments += toolCall.Function.Arguments
						}
					}
				}

				// Do not break on the finish reason: with StreamOptions.IncludeUsage
				// the usage arrives in a trailing chunk (empty choices) after the
				// finish-reason chunk. Keep reading until io.EOF so token usage
				// (input/output and cache reads) is captured.
			}

			// Process accumulated tool calls
			if len(toolCalls) > 0 {
				var functionCalls []*gollem.FunctionCall
				for _, toolCall := range toolCalls {
					if toolCall.ID != "" && toolCall.Function.Name != "" && toolCall.Function.Arguments != "" {
						args, err := jsonutil.DecodeObject([]byte(toolCall.Function.Arguments))
						if err != nil {
							responseChan <- &gollem.ContentResponse{
								Error: goerr.Wrap(err, "failed to unmarshal function call arguments"),
							}
							return
						}

						functionCalls = append(functionCalls, &gollem.FunctionCall{
							ID:        toolCall.ID,
							Name:      toolCall.Function.Name,
							Arguments: args,
						})
					}
				}

				if len(functionCalls) > 0 {
					responseChan <- &gollem.ContentResponse{
						FunctionCalls:       functionCalls,
						InputToken:          totalInputTokens,
						OutputToken:         totalOutputTokens,
						CacheReadInputToken: totalCacheRead,
					}
				}

				// Create assistant message with tool calls
				assistantMessage := openai.ChatCompletionMessage{
					Role:      openai.ChatMessageRoleAssistant,
					ToolCalls: toolCalls,
				}
				// Update history with assistant response
				if err := s.updateHistoryWithResponse(assistantMessage); err != nil {
					responseChan <- &gollem.ContentResponse{
						Error: goerr.Wrap(err, "failed to update history with assistant response"),
					}
					return
				}
			} else if textContent != "" || reasoningContent != "" {
				// Create assistant message with text and/or reasoning content
				assistantMessage := openai.ChatCompletionMessage{
					Role:             openai.ChatMessageRoleAssistant,
					Content:          textContent,
					ReasoningContent: reasoningContent,
				}
				// Update history with assistant response
				if err := s.updateHistoryWithResponse(assistantMessage); err != nil {
					responseChan <- &gollem.ContentResponse{
						Error: goerr.Wrap(err, "failed to update history with assistant response"),
					}
					return
				}
			}

			// Set trace data for defer.
			// Record only messages added in this turn; previous turns are already
			// captured in earlier trace spans.
			streamTraceData = &trace.LLMCallData{
				InputTokens:          totalInputTokens,
				OutputTokens:         totalOutputTokens,
				Model:                s.defaultModel,
				CacheReadInputTokens: totalCacheRead,
				Request: &trace.LLMRequest{
					SystemPrompt: s.cfg.SystemPrompt(),
					Messages:     openaiMessagesToTraceMessages(newMessages),
				},
				Response: &trace.LLMResponse{},
			}
			if textContent != "" {
				streamTraceData.Response.Texts = []string{textContent}
			}
			for _, tc := range toolCalls {
				if tc.ID != "" && tc.Function.Name != "" {
					args := traceArguments(tc.Function.Arguments)
					streamTraceData.Response.FunctionCalls = append(streamTraceData.Response.FunctionCalls, &trace.FunctionCall{
						ID:        tc.ID,
						Name:      tc.Function.Name,
						Arguments: args,
					})
				}
			}

			// Send final response with complete token usage if available
			if totalInputTokens > 0 || totalOutputTokens > 0 {
				responseChan <- &gollem.ContentResponse{
					InputToken:          totalInputTokens,
					OutputToken:         totalOutputTokens,
					CacheReadInputToken: totalCacheRead,
				}
			}

			// History is already updated by updateHistoryWithResponse above
		}()

		return responseChan, nil
	}

	// Build middleware chain
	handler := gollem.ContentStreamHandler(baseHandler)
	for i := len(s.cfg.ContentStreamMiddlewares()) - 1; i >= 0; i-- {
		handler = s.cfg.ContentStreamMiddlewares()[i](handler)
	}

	// Execute middleware chain
	streamChan, err := handler(ctx, contentReq)
	if err != nil {
		return nil, err
	}

	// Sanity check: streamChan should not be nil if err is nil
	if streamChan == nil {
		return nil, goerr.New("middleware returned nil channel without error")
	}

	// Convert ContentStreamResponse channel to Response channel
	responseChan := make(chan *gollem.Response)
	go func() {
		defer close(responseChan)
		for streamResp := range streamChan {
			if streamResp.Error != nil {
				responseChan <- &gollem.Response{
					Error: streamResp.Error,
				}
			} else {
				responseChan <- &gollem.Response{
					Texts:               streamResp.Texts,
					Thoughts:            streamResp.Thoughts,
					FunctionCalls:       streamResp.FunctionCalls,
					InputToken:          streamResp.InputToken,
					OutputToken:         streamResp.OutputToken,
					CacheReadInputToken: streamResp.CacheReadInputToken,
				}
			}
		}
	}()

	return responseChan, nil
}

// convertResponseSchemaToOpenAI converts gollem.ResponseSchema to OpenAI's JSONSchemaParams
func convertResponseSchemaToOpenAI(param *gollem.Parameter, strict bool) (*openai.ChatCompletionResponseFormatJSONSchema, error) {
	if param == nil {
		return nil, nil
	}

	// Validate schema
	if err := param.Validate(); err != nil {
		return nil, goerr.Wrap(err, "invalid response schema")
	}

	// Convert Parameter to JSON Schema format
	// If strict mode is enabled, we need to adjust the schema to make all properties required
	// This is a limitation of OpenAI's strict mode implementation
	schemaObj := convertParameterToJSONSchemaWithStrict(param, strict)

	// Marshal to JSON for OpenAI API
	schemaJSON, err := json.Marshal(schemaObj)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to marshal schema")
	}

	name := param.Title
	if name == "" {
		name = "response"
	}

	result := &openai.ChatCompletionResponseFormatJSONSchema{
		Name:        name,
		Description: param.Description,
		Schema:      json.RawMessage(schemaJSON),
		Strict:      strict,
	}

	return result, nil
}

// convertParameterToJSONSchemaWithStrict converts gollem.Parameter to JSON Schema map
// with optional strict mode handling for OpenAI
func convertParameterToJSONSchemaWithStrict(param *gollem.Parameter, strict bool) map[string]any {
	// For non-strict mode, use the shared conversion function
	if !strict {
		return schema.ConvertParameterToJSONSchema(param)
	}

	// Strict mode: OpenAI-specific handling
	// In strict mode, all properties must be in the required array
	result := map[string]any{
		"type": string(param.Type),
	}

	if param.Description != "" {
		result["description"] = param.Description
	}

	if param.Type == gollem.TypeObject && param.Properties != nil {
		props := make(map[string]any)
		for name, prop := range param.Properties {
			props[name] = convertParameterToJSONSchemaWithStrict(prop, strict)
		}
		result["properties"] = props
		result["additionalProperties"] = false

		// In strict mode, OpenAI requires all properties to be in the required array
		// This is a limitation of OpenAI's strict mode, not a general JSON Schema requirement
		allKeys := make([]string, 0, len(param.Properties))
		for key := range param.Properties {
			allKeys = append(allKeys, key)
		}
		// Sort so the emitted schema is byte-identical between identical requests
		slices.Sort(allKeys)
		result["required"] = allKeys
	}

	if param.Type == gollem.TypeArray && param.Items != nil {
		result["items"] = convertParameterToJSONSchemaWithStrict(param.Items, strict)
	}

	if param.Enum != nil {
		result["enum"] = param.Enum
	}

	// Add constraints
	if param.Minimum != nil {
		result["minimum"] = *param.Minimum
	}
	if param.Maximum != nil {
		result["maximum"] = *param.Maximum
	}
	if param.MinLength != nil {
		result["minLength"] = *param.MinLength
	}
	if param.MaxLength != nil {
		result["maxLength"] = *param.MaxLength
	}
	if param.Pattern != "" {
		result["pattern"] = param.Pattern
	}
	if param.MinItems != nil {
		result["minItems"] = *param.MinItems
	}
	if param.MaxItems != nil {
		result["maxItems"] = *param.MaxItems
	}

	return result
}

// applyPerCallOverrides applies per-call GenerateOption overrides to an API request.
func (s *Session) applyPerCallOverrides(req *openai.ChatCompletionRequest, opts ...gollem.GenerateOption) error {
	genCfg := gollem.NewGenerateConfig(opts...)
	if t := genCfg.Temperature(); t != nil {
		req.Temperature = float32(*t)
	}
	if p := genCfg.TopP(); p != nil {
		req.TopP = float32(*p)
	}
	if m := genCfg.MaxTokens(); m != nil {
		req.MaxCompletionTokens = *m
	}
	if schema := genCfg.ResponseSchema(); schema != nil {
		jsonSchema, err := convertResponseSchemaToOpenAI(schema, s.strictMode)
		if err != nil {
			return goerr.Wrap(err, "failed to convert per-call response schema")
		}
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type:       openai.ChatCompletionResponseFormatTypeJSONSchema,
			JSONSchema: jsonSchema,
		}
	}
	return nil
}

// Deprecated: GenerateContent is deprecated. Use Generate instead.
func (s *Session) GenerateContent(ctx context.Context, input ...gollem.Input) (*gollem.Response, error) {
	return s.Generate(ctx, input)
}

// Deprecated: GenerateStream is deprecated. Use Stream instead.
func (s *Session) GenerateStream(ctx context.Context, input ...gollem.Input) (<-chan *gollem.Response, error) {
	return s.Stream(ctx, input)
}

// CountToken calculates the total number of tokens for the given inputs,
// including system prompt, history messages, and new inputs.
// This uses tiktoken library for local token counting without API calls.
func (s *Session) CountToken(ctx context.Context, input ...gollem.Input) (int, error) {
	// Get tiktoken encoding for the model
	// If model is not found, try to use a compatible encoding
	encoding, err := tiktoken.EncodingForModel(s.defaultModel)
	if err != nil {
		// Fallback to cl100k_base encoding (used by gpt-4, gpt-3.5-turbo, gpt-4o, gpt-5, etc.)
		encoding, err = tiktoken.GetEncoding("cl100k_base")
		if err != nil {
			return 0, goerr.Wrap(err, "failed to get encoding")
		}
	}

	// Convert inputs to messages without modifying session state
	newMessages, err := s.convertInputsToMessages(input...)
	if err != nil {
		return 0, goerr.Wrap(err, "failed to convert inputs for token counting")
	}

	// Create a copy of history messages to avoid race conditions
	// This ensures thread safety when reading historyMessages
	historyMessagesCopy := make([]openai.ChatCompletionMessage, len(s.historyMessages))
	copy(historyMessagesCopy, s.historyMessages)

	// Combine history copy with new inputs for counting
	messages := append(historyMessagesCopy, newMessages...)

	// Count tokens for all messages
	totalTokens := 0

	// Add tokens for system prompt if present
	if s.cfg.SystemPrompt() != "" {
		totalTokens += len(encoding.Encode(s.cfg.SystemPrompt(), nil, nil))
		totalTokens += 3 // System message formatting tokens
	}

	// Count tokens per message based on model
	// Different models have different token overhead per message
	tokensPerMessage := 3
	tokensPerName := 1

	// Adjust for specific model families
	switch s.defaultModel {
	case "gpt-3.5-turbo-0301":
		tokensPerMessage = 4
		tokensPerName = -1
	}

	for _, message := range messages {
		totalTokens += tokensPerMessage
		if message.Content != "" {
			totalTokens += len(encoding.Encode(message.Content, nil, nil))
		}
		totalTokens += len(encoding.Encode(message.Role, nil, nil))
		if message.Name != "" {
			totalTokens += len(encoding.Encode(message.Name, nil, nil))
			totalTokens += tokensPerName
		}
		// Count tool calls
		if message.ToolCalls != nil {
			for _, toolCall := range message.ToolCalls {
				totalTokens += len(encoding.Encode(toolCall.Function.Name, nil, nil))
				totalTokens += len(encoding.Encode(toolCall.Function.Arguments, nil, nil))
			}
		}
		// Count multi-content parts
		if message.MultiContent != nil {
			for _, part := range message.MultiContent {
				if part.Type == openai.ChatMessagePartTypeText {
					totalTokens += len(encoding.Encode(part.Text, nil, nil))
				}
			}
		}
	}

	// Add tokens for tools if present
	// Create a copy of tools to avoid race conditions
	toolsCopy := make([]openai.Tool, len(s.tools))
	copy(toolsCopy, s.tools)

	if len(toolsCopy) > 0 {
		for _, tool := range toolsCopy {
			toolJSON, err := json.Marshal(tool)
			if err != nil {
				return 0, goerr.Wrap(err, "failed to marshal tool for token counting")
			}
			totalTokens += len(encoding.Encode(string(toolJSON), nil, nil))
		}
	}

	// Add reply priming tokens
	totalTokens += 3

	return totalTokens, nil
}

// tokenLimitErrorOptions checks if the error is a token limit exceeded error
// and returns goerr.Option to tag the error with ErrTagTokenExceeded.
// Returns nil if the error is not a token limit exceeded error.
//
// Detection logic:
// - Error must be *openai.APIError
// - Type must be "invalid_request_error"
// - Code must be "context_length_exceeded" (as string)
func tokenLimitErrorOptions(err error) []goerr.Option {
	var apiErr *openai.APIError
	if !errors.As(err, &apiErr) {
		return nil
	}

	if apiErr.Type != "invalid_request_error" {
		return nil
	}

	codeStr, ok := apiErr.Code.(string)
	if !ok {
		return nil
	}

	if codeStr == "context_length_exceeded" {
		return []goerr.Option{goerr.Tag(gollem.ErrTagTokenExceeded)}
	}

	return nil
}

// traceArguments decodes a function call's arguments for trace output. Trace data is
// diagnostic and must never fail the request that produced it, so arguments that do not
// parse as a JSON object are recorded verbatim under "arguments" rather than dropped.
func traceArguments(arguments string) map[string]any {
	if arguments == "" {
		return nil
	}
	args, err := jsonutil.DecodeObject([]byte(arguments))
	if err != nil {
		return map[string]any{rawArgumentsKey: arguments}
	}
	return args
}

// openaiMessagesToTraceMessages converts OpenAI messages to trace messages.
func openaiMessagesToTraceMessages(messages []openai.ChatCompletionMessage) []trace.Message {
	var result []trace.Message
	for _, msg := range messages {
		var blocks []trace.MessageContent
		if msg.ToolCallID != "" {
			// Tool response message: combine content into the tool_response block
			tc := trace.NewToolResponseContent(msg.ToolCallID, msg.Name, nil)
			if msg.Content != "" {
				tc.Text = msg.Content
			}
			blocks = append(blocks, tc)
		} else {
			if msg.Content != "" {
				blocks = append(blocks, trace.NewTextContent(msg.Content))
			}
			for _, mc := range msg.MultiContent {
				switch mc.Type {
				case openai.ChatMessagePartTypeText:
					if mc.Text != "" {
						blocks = append(blocks, trace.NewTextContent(mc.Text))
					}
				case openai.ChatMessagePartTypeImageURL:
					imgBlock := trace.NewMediaContent("image", "")
					if mc.ImageURL != nil {
						imgBlock.URL = mc.ImageURL.URL
					}
					blocks = append(blocks, imgBlock)
				}
			}
			for _, tc := range msg.ToolCalls {
				args := traceArguments(tc.Function.Arguments)
				blocks = append(blocks, trace.NewToolCallContent(
					tc.ID, tc.Function.Name, args,
				))
			}
		}
		if len(blocks) > 0 {
			result = append(result, trace.Message{
				Role:     msg.Role,
				Contents: blocks,
			})
		}
	}
	return result
}

// buildOpenAITraceData builds trace.LLMCallData from an OpenAI API response.
func buildOpenAITraceData(resp openai.ChatCompletionResponse, model string, systemPrompt string, messages []openai.ChatCompletionMessage) *trace.LLMCallData {
	data := &trace.LLMCallData{
		InputTokens:          resp.Usage.PromptTokens,
		OutputTokens:         resp.Usage.CompletionTokens,
		Model:                resp.Model,
		CacheReadInputTokens: cachedPromptTokens(resp.Usage),
		Request: &trace.LLMRequest{
			SystemPrompt: systemPrompt,
			Messages:     openaiMessagesToTraceMessages(messages),
		},
		Response: &trace.LLMResponse{},
	}

	if len(resp.Choices) > 0 {
		message := resp.Choices[0].Message
		if message.Content != "" {
			data.Response.Texts = append(data.Response.Texts, message.Content)
		}
		for _, toolCall := range message.ToolCalls {
			var args map[string]any
			if toolCall.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
					args = map[string]any{
						"__raw_arguments": toolCall.Function.Arguments,
						"__error":         err.Error(),
					}
				}
			}
			data.Response.FunctionCalls = append(data.Response.FunctionCalls, &trace.FunctionCall{
				ID:        toolCall.ID,
				Name:      toolCall.Function.Name,
				Arguments: args,
			})
		}
	}

	return data
}
