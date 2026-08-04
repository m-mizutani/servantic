package claude

import (
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

// Export convert functions for testing
var (
	ConvertTool                   = convertTool
	ConvertParameterToSchema      = convertParameterToSchema
	ConvertGollemInputsToClaude   = convertGollemInputsToClaude
	CreateSystemPrompt            = createSystemPrompt
	TokenLimitErrorOptions        = tokenLimitErrorOptions
	ClaudeMessagesToTraceMessages = claudeMessagesToTraceMessages
	ApplyPromptCacheBreakpoints   = applyPromptCacheBreakpoints
	CacheTokensFromUsage          = cacheTokensFromUsage
)

type JsonSchema = jsonSchema

// Export for testing
type APIClient = apiClient

// NewSessionWithAPIClient creates a new session with a custom API client for testing
func NewSessionWithAPIClient(client apiClient, cfg gollem.SessionConfig, model string) (*Session, error) {
	tools := make([]anthropic.ToolUnionParam, 0, len(cfg.Tools()))
	for _, tool := range cfg.Tools() {
		tools = append(tools, convertTool(tool))
	}

	// Initialize historyMessages from config
	var historyMessages []anthropic.MessageParam
	if cfg.History() != nil {
		var err error
		historyMessages, err = ToMessages(cfg.History())
		if err != nil {
			return nil, err
		}
	}

	return &Session{
		apiClient:       client,
		defaultModel:    model,
		tools:           tools,
		historyMessages: historyMessages,
		params: generationParameters{
			Temperature: -1.0,
			TopP:        -1.0,
			MaxTokens:   8192,
		},
		cfg: cfg,
	}, nil
}

// NewVertexSessionWithClient creates a Vertex session backed by the given Anthropic
// client for testing. NewWithVertex installs Google auth on its client, so tests
// that need to drive the Vertex code path against an httptest server must supply
// their own client here.
func NewVertexSessionWithClient(client *anthropic.Client, cfg gollem.SessionConfig, model string) (*VertexAnthropicSession, error) {
	var messages []anthropic.MessageParam
	if cfg.History() != nil {
		history, err := ToMessages(cfg.History())
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert history to anthropic.MessageParam")
		}
		messages = append(messages, history...)
	}

	return &VertexAnthropicSession{
		client:       client,
		defaultModel: model,
		params: generationParameters{
			Temperature: -1.0,
			TopP:        -1.0,
			MaxTokens:   8192,
		},
		cfg:      cfg,
		messages: messages,
	}, nil
}

// GetBaseURL returns the base URL from a Claude client for testing
func GetBaseURL(client *Client) string {
	return client.baseURL
}
