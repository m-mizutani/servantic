package gemini

import (
	"github.com/gollem-dev/gollem"
	"google.golang.org/genai"
)

// Export convert functions for testing
var (
	ConvertTool                = convertTool
	ConvertParameterToSchema   = convertParameterToSchema
	TokenLimitErrorOptions     = tokenLimitErrorOptions
	ContentsToTraceMessages    = contentsToTraceMessages
	GeminiFallbackToolCallID   = geminiFallbackToolCallID
	IsGeminiFallbackToolCallID = isGeminiFallbackToolCallID
	ProcessResponse            = processResponse
)

// GetGenerationConfig returns the generationConfig for testing
func (c *Client) GetGenerationConfig() *genai.GenerateContentConfig {
	return c.generationConfig
}

// NewClientWithOptions builds a Client through the same defaults and option
// handling New uses, for tests that must not reach GCP.
func NewClientWithOptions(options ...Option) *Client {
	return newConfiguredClient("test-project", "test-location", options...)
}

// Export for testing
type APIClient = apiClient

// NewSessionWithAPIClient creates a new session with a custom API client for testing
func NewSessionWithAPIClient(client apiClient, cfg gollem.SessionConfig, model string) (*Session, error) {
	// Initialize historyContents from config
	var historyContents []*genai.Content
	if cfg.History() != nil {
		var err error
		historyContents, err = ToContents(cfg.History())
		if err != nil {
			return nil, err
		}
	}

	// Create generation config
	config := &genai.GenerateContentConfig{}

	return &Session{
		apiClient:       client,
		model:           model,
		config:          config,
		historyContents: historyContents,
		cfg:             cfg,
	}, nil
}

// SetSessionAPIClient sets the API client for testing
func SetSessionAPIClient(s *Session, client apiClient) {
	s.apiClient = client
}

// SetSessionModel sets the model for testing
func SetSessionModel(s *Session, model string) {
	s.model = model
}

// SetSessionConfig sets the config for testing
func SetSessionConfig(s *Session, config *genai.GenerateContentConfig) {
	s.config = config
}

// SetSessionCfg sets the cfg for testing
func SetSessionCfg(s *Session, cfg gollem.SessionConfig) {
	s.cfg = cfg
}
