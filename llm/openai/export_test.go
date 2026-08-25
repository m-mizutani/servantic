package openai

import (
	"github.com/gollem-dev/gollem"
	"github.com/sashabaranov/go-openai"
)

// Export convert functions for testing
var (
	ConvertTool                   = convertTool
	ConvertParameterToSchema      = convertParameterToSchema
	ConvertResponseSchemaToOpenAI = convertResponseSchemaToOpenAI
	TokenLimitErrorOptions        = tokenLimitErrorOptions
	OpenaiMessagesToTraceMessages = openaiMessagesToTraceMessages
)

// Export for testing
type APIClient = apiClient

// NewSessionWithAPIClient creates a new session with a custom API client for testing
func NewSessionWithAPIClient(client apiClient, cfg gollem.SessionConfig, model string) (*Session, error) {
	tools := make([]openai.Tool, 0, len(cfg.Tools()))
	for _, tool := range cfg.Tools() {
		tools = append(tools, convertTool(tool))
	}

	// Initialize historyMessages from config
	var historyMessages []openai.ChatCompletionMessage
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
		params:          generationParameters{},
		cfg:             cfg,
	}, nil
}

// GetBaseURL returns the base URL from an OpenAI client for testing
func GetBaseURL(client *Client) string {
	return client.baseURL
}
