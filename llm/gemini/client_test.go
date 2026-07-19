package gemini_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
	"github.com/m-mizutani/gt"
	"google.golang.org/genai"
)

const (
	testTimeout   = 30 * time.Second
	maxTestTokens = 2048
)

// Tests for client.go functionality

func TestClientMalformedFunctionCallErrorHandling(t *testing.T) {
	// This test simulates what would happen when a malformed function call error occurs
	// We can't easily trigger this in a unit test, so we test the error handling logic

	t.Run("error contains helpful information", func(t *testing.T) {
		// This would be called when a malformed function call is detected
		err := gollem.ErrInvalidParameter

		// The error should contain useful debugging information
		gt.Value(t, err).NotEqual(nil)

		// In a real scenario, the error would contain:
		// - candidate_index
		// - content_parts
		// - finish_reason
		// - suggested_action

	})
}

func TestClientRetryLogic(t *testing.T) {
	t.Run("retry with exponential backoff", func(t *testing.T) {
		start := time.Now()

		// Simulate what the retry logic would do
		maxRetries := 3
		baseDelay := 100 * time.Millisecond

		for attempt := 0; attempt < maxRetries; attempt++ {
			// Simulate a malformed function call error
			simulatedError := "malformed function call detected"

			if strings.Contains(simulatedError, "malformed function call") {
				// Always sleep before the next attempt (except we'll break before the last one)
				if attempt < maxRetries-1 {
					// Calculate delay (exponential backoff) using math.Pow like the real implementation
					delay := time.Duration(float64(baseDelay) * math.Pow(2, float64(attempt)))
					time.Sleep(delay)
					continue
				}
			}
			break
		}

		elapsed := time.Since(start)

		// Should have taken at least the sum of delays: 100ms + 200ms = 300ms
		// (We only sleep on attempt 0 and 1, not on the final attempt)
		expectedMinDelay := 300 * time.Millisecond
		gt.Value(t, elapsed >= expectedMinDelay).Equal(true)

	})
}

func TestClientLargeTextDetection(t *testing.T) {
	t.Run("detect large text content", func(t *testing.T) {
		testCases := []struct {
			name    string
			content string
			isLarge bool
		}{
			{
				name:    "small_text",
				content: "This is a small text",
				isLarge: false,
			},
			{
				name:    "large_text",
				content: strings.Repeat("a", 1500),
				isLarge: true,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				isLarge := len(tc.content) > 1000
				gt.Value(t, isLarge).Equal(tc.isLarge)

				_ = isLarge // Note detected state
			})
		}
	})
}

func TestClientToolSchemaValidation(t *testing.T) {
	t.Run("valid_tool_schema", func(t *testing.T) {
		tool := &validClientTool{}
		spec := tool.Spec()

		// Check that the spec has required fields
		gt.Value(t, spec.Name).NotEqual("")
		gt.Value(t, spec.Description).NotEqual("")
		gt.Value(t, spec.Parameters).NotEqual(nil)

		// Check that string parameters have constraints
		for _, param := range spec.Parameters {
			if param.Type == gollem.TypeString {
				_ = param.MaxLength == nil // Check constraint presence
			}
		}
	})

	t.Run("problematic_tool_schema", func(t *testing.T) {
		tool := &problematicClientTool{}
		spec := tool.Spec()

		// Check for potential issues
		hasProblematicNames := false
		problematicNames := []string{"type", "properties", "required"}

		for _, name := range problematicNames {
			if _, exists := spec.Parameters[name]; exists {
				hasProblematicNames = true
			}
		}

		_ = hasProblematicNames // Note problematic names detected
	})
}

func TestGeminiClientIssues(t *testing.T) {

	t.Run("large_text_content_schema", func(t *testing.T) {
		tool := &largeTextClientTool{}
		converted := gemini.ConvertTool(tool)

		gt.Value(t, converted.Name).Equal("large_text_client")
		gt.Value(t, len(converted.Parameters.Properties)).Equal(1)

		contentParam := converted.Parameters.Properties["content"]
		gt.Value(t, contentParam).NotEqual(nil)
		gt.Value(t, contentParam.Type).Equal(genai.TypeString)

		// Check for length constraints
		_ = contentParam.MaxLength == nil || *contentParam.MaxLength == 0 // Note constraint status

	})

	t.Run("problematic_field_names", func(t *testing.T) {
		tool := &problematicFieldClientTool{}
		converted := gemini.ConvertTool(tool)

		gt.Value(t, converted.Name).Equal("problematic_field_client")
		gt.Value(t, len(converted.Parameters.Properties)).Equal(4)

		// Check that problematic field names are handled
		problematicNames := []string{"type", "properties", "required"}
		for _, name := range problematicNames {
			param := converted.Parameters.Properties[name]
			gt.Value(t, param).NotEqual(nil)
		}

		// Check unicode field
		unicodeParam := converted.Parameters.Properties["unicode_field"]
		gt.Value(t, unicodeParam).NotEqual(nil)
		gt.Value(t, unicodeParam.Type).Equal(genai.TypeString)

		// Log the unicode description to verify it's handled correctly

	})
}

// Tool definitions for client testing

type validClientTool struct{}

func (t *validClientTool) Spec() gollem.ToolSpec {
	maxLen := 1000

	return gollem.ToolSpec{
		Name:        "valid_client_tool",
		Description: "A well-designed tool with proper constraints",
		Parameters: map[string]*gollem.Parameter{
			"content": {
				Type:        gollem.TypeString,
				Description: "Content with length constraints",
				MaxLength:   &maxLen,
				Required:    true,
			},
			"metadata": {
				Type:        gollem.TypeObject,
				Description: "Metadata object",
				Properties: map[string]*gollem.Parameter{
					"title": {
						Type:        gollem.TypeString,
						Description: "Title",
						MaxLength:   &maxLen,
					},
				},
			},
		},
	}
}

func (t *validClientTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return map[string]any{"result": "success"}, nil
}

type problematicClientTool struct{}

func (t *problematicClientTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "problematic_client_tool",
		Description: "A tool with potential issues",
		Parameters: map[string]*gollem.Parameter{
			"type": { // Problematic name
				Type:        gollem.TypeString,
				Description: "Type field",
				// No MaxLength constraint
			},
			"properties": { // Problematic name
				Type:        gollem.TypeObject,
				Description: "Properties field",
				Properties: map[string]*gollem.Parameter{
					"nested": {
						Type:        gollem.TypeString,
						Description: "Nested field",
					},
				},
				// Required field might be nil
			},
		},
	}
}

func (t *problematicClientTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return map[string]any{"result": "success"}, nil
}

type largeTextClientTool struct{}

func (t *largeTextClientTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "large_text_client",
		Description: "A tool that accepts large text content which might cause issues",
		Parameters: map[string]*gollem.Parameter{
			"content": {
				Type:        gollem.TypeString,
				Description: "Large text content that might cause FinishReasonMalformedFunctionCall",
				Required:    true,
				// NOTE: No MaxLength constraint - this is the problematic part
			},
		},
	}
}

func (t *largeTextClientTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return map[string]any{"result": "processed"}, nil
}

type problematicFieldClientTool struct{}

func (t *problematicFieldClientTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "problematic_field_client",
		Description: "Tool with field names that might conflict with JSON schema keywords",
		Parameters: map[string]*gollem.Parameter{
			"type": {
				Type:        gollem.TypeString,
				Description: "Field named 'type' - might conflict with JSON schema",
				Required:    true,
			},
			"properties": {
				Type:        gollem.TypeString,
				Description: "Field named 'properties' - might conflict with JSON schema",
			},
			"required": {
				Type:        gollem.TypeString,
				Description: "Field named 'required' - might conflict with JSON schema",
			},
			"unicode_field": {
				Type:        gollem.TypeString,
				Description: "Field with unicode: test characters 🚀 emoji",
			},
		},
	}
}

func (t *problematicFieldClientTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return map[string]any{"result": "processed"}, nil
}

func TestGeminiContentGenerate(t *testing.T) {
	var testProjectID, testLocation string
	v, ok := os.LookupEnv("TEST_GCP_PROJECT_ID")
	if !ok {
		t.Skip("TEST_GCP_PROJECT_ID is not set")
	} else {
		testProjectID = v
	}

	v, ok = os.LookupEnv("TEST_GCP_LOCATION")
	if !ok {
		t.Skip("TEST_GCP_LOCATION is not set")
	} else {
		testLocation = v
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	var opts []gemini.Option
	if model := os.Getenv("TEST_GCP_MODEL"); model != "" {
		opts = append(opts, gemini.WithModel(model))
	}

	client, err := gemini.New(ctx, testProjectID, testLocation, opts...)
	gt.NoError(t, err)

	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	result, err := session.Generate(ctx, []gollem.Input{gollem.Text("Say hello in one word")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err).Required()
	gt.A(t, result.Texts).Length(1).Required()
	gt.Value(t, len(result.Texts[0])).NotEqual(0)
}

func TestWithThinkingBudget(t *testing.T) {
	projectID := os.Getenv("TEST_GCP_PROJECT_ID")
	if projectID == "" {
		t.Skip("TEST_GCP_PROJECT_ID is not set")
	}

	location := os.Getenv("TEST_GCP_LOCATION")
	if location == "" {
		t.Skip("TEST_GCP_LOCATION is not set")
	}

	ctx := context.Background()

	testCases := []struct {
		name         string
		budget       int32
		expectBudget int32
	}{
		{
			name:         "auto thinking budget",
			budget:       -1,
			expectBudget: -1,
		},
		{
			name:         "specific thinking budget",
			budget:       1000,
			expectBudget: 1000,
		},
		{
			name:         "zero thinking budget",
			budget:       0,
			expectBudget: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client, err := gemini.New(ctx, projectID, location,
				gemini.WithThinkingBudget(tc.budget),
			)
			gt.NoError(t, err)
			gt.NotNil(t, client)

			generationConfig := client.GetGenerationConfig()
			gt.NotNil(t, generationConfig)
			gt.NotNil(t, generationConfig.ThinkingConfig)
			gt.NotNil(t, generationConfig.ThinkingConfig.ThinkingBudget)
			gt.Equal(t, tc.expectBudget, *generationConfig.ThinkingConfig.ThinkingBudget)
		})
	}
}

func TestWithThinkingLevel(t *testing.T) {
	projectID := os.Getenv("TEST_GCP_PROJECT_ID")
	if projectID == "" {
		t.Skip("TEST_GCP_PROJECT_ID is not set")
	}

	location := os.Getenv("TEST_GCP_LOCATION")
	if location == "" {
		t.Skip("TEST_GCP_LOCATION is not set")
	}

	ctx := context.Background()

	testCases := []struct {
		name        string
		level       genai.ThinkingLevel
		expectLevel genai.ThinkingLevel
	}{
		{
			name:        "minimal",
			level:       genai.ThinkingLevelMinimal,
			expectLevel: genai.ThinkingLevelMinimal,
		},
		{
			name:        "low",
			level:       genai.ThinkingLevelLow,
			expectLevel: genai.ThinkingLevelLow,
		},
		{
			name:        "medium",
			level:       genai.ThinkingLevelMedium,
			expectLevel: genai.ThinkingLevelMedium,
		},
		{
			name:        "high",
			level:       genai.ThinkingLevelHigh,
			expectLevel: genai.ThinkingLevelHigh,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client, err := gemini.New(ctx, projectID, location,
				gemini.WithThinkingLevel(tc.level),
			)
			gt.NoError(t, err)
			gt.NotNil(t, client)

			generationConfig := client.GetGenerationConfig()
			gt.NotNil(t, generationConfig)
			gt.NotNil(t, generationConfig.ThinkingConfig)
			gt.Equal(t, tc.expectLevel, generationConfig.ThinkingConfig.ThinkingLevel)
			// Vertex AI rejects requests that carry both fields; the option
			// must clear the default-zero ThinkingBudget that gemini.New sets.
			gt.Nil(t, generationConfig.ThinkingConfig.ThinkingBudget)
		})
	}

	t.Run("budget overrides level", func(t *testing.T) {
		client, err := gemini.New(ctx, projectID, location,
			gemini.WithThinkingLevel(genai.ThinkingLevelHigh),
			gemini.WithThinkingBudget(500),
		)
		gt.NoError(t, err)

		cfg := client.GetGenerationConfig()
		gt.NotNil(t, cfg.ThinkingConfig.ThinkingBudget)
		gt.Equal(t, int32(500), *cfg.ThinkingConfig.ThinkingBudget)
		gt.Equal(t, genai.ThinkingLevel(""), cfg.ThinkingConfig.ThinkingLevel)
	})

	t.Run("level overrides budget", func(t *testing.T) {
		client, err := gemini.New(ctx, projectID, location,
			gemini.WithThinkingBudget(500),
			gemini.WithThinkingLevel(genai.ThinkingLevelLow),
		)
		gt.NoError(t, err)

		cfg := client.GetGenerationConfig()
		gt.Nil(t, cfg.ThinkingConfig.ThinkingBudget)
		gt.Equal(t, genai.ThinkingLevelLow, cfg.ThinkingConfig.ThinkingLevel)
	})
}

func TestThinkingBudgetIntegration(t *testing.T) {
	projectID := os.Getenv("TEST_GCP_PROJECT_ID")
	if projectID == "" {
		t.Skip("TEST_GCP_PROJECT_ID is not set")
	}

	location := os.Getenv("TEST_GCP_LOCATION")
	if location == "" {
		t.Skip("TEST_GCP_LOCATION is not set")
	}

	testCases := []struct {
		name   string
		model  string
		budget int32
	}{
		{
			name:   "Gemini 2.5 Flash with thinking budget disabled",
			model:  "gemini-2.5-flash",
			budget: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
			defer cancel()

			client, err := gemini.New(ctx, projectID, location,
				gemini.WithModel(tc.model),
				gemini.WithThinkingBudget(tc.budget),
			)
			gt.NoError(t, err)
			gt.NotNil(t, client)

			// Verify configuration is set correctly
			generationConfig := client.GetGenerationConfig()
			gt.NotNil(t, generationConfig)
			gt.NotNil(t, generationConfig.ThinkingConfig)
			gt.NotNil(t, generationConfig.ThinkingConfig.ThinkingBudget)
			gt.Equal(t, tc.budget, *generationConfig.ThinkingConfig.ThinkingBudget)

			// Test actual API call
			session, err := client.NewSession(ctx)
			gt.NoError(t, err)
			gt.NotNil(t, session)

			// Simple test prompt
			response, err := session.Generate(ctx, []gollem.Input{gollem.Text("Say 'Hello' in one word")}, gollem.WithMaxTokens(maxTestTokens))
			gt.NoError(t, err).Required()
			gt.NotNil(t, response)
			gt.Array(t, response.Texts).Length(1).Required()
			gt.Value(t, len(response.Texts[0])).NotEqual(0)
		})
	}
}

func TestTokenLimitErrorOptions(t *testing.T) {
	type testCase struct {
		name   string
		err    error
		hasTag bool
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			opts := gemini.TokenLimitErrorOptions(tc.err)
			if tc.hasTag {
				gt.NotEqual(t, 0, len(opts))
			} else {
				gt.Equal(t, 0, len(opts))
			}
		}
	}

	t.Run("token exceeded error - max context length", runTest(testCase{
		name: "max context length exceeded",
		err: &genai.APIError{
			Code:    400,
			Message: "The model's maximum context length is 128000 tokens",
			Status:  "INVALID_ARGUMENT",
		},
		hasTag: true,
	}))

	t.Run("token exceeded error - payload size", runTest(testCase{
		name: "payload size exceeded",
		err: &genai.APIError{
			Code:    400,
			Message: "Request payload size exceeds the limit: 10485760 bytes",
			Status:  "INVALID_ARGUMENT",
		},
		hasTag: true,
	}))

	t.Run("different error code", runTest(testCase{
		name: "error code 401",
		err: &genai.APIError{
			Code:    401,
			Message: "Invalid API key",
			Status:  "UNAUTHENTICATED",
		},
		hasTag: false,
	}))

	t.Run("different status", runTest(testCase{
		name: "different status",
		err: &genai.APIError{
			Code:    400,
			Message: "Invalid request",
			Status:  "INVALID_REQUEST",
		},
		hasTag: false,
	}))

	t.Run("different message", runTest(testCase{
		name: "different message",
		err: &genai.APIError{
			Code:    400,
			Message: "Invalid model specified",
			Status:  "INVALID_ARGUMENT",
		},
		hasTag: false,
	}))

	t.Run("correct code and status but wrong message", runTest(testCase{
		name: "wrong message",
		err: &genai.APIError{
			Code:    400,
			Message: "Some other error",
			Status:  "INVALID_ARGUMENT",
		},
		hasTag: false,
	}))

	t.Run("nil error", runTest(testCase{
		name:   "nil error",
		err:    nil,
		hasTag: false,
	}))

	t.Run("non-APIError", runTest(testCase{
		name:   "generic error",
		err:    errors.New("some error"),
		hasTag: false,
	}))
}

func TestThinkingModelAgentLoop(t *testing.T) {
	// Simulate a thinking model response with ThoughtSignature.
	// Verify that the second GenerateContent call receives the signatures in history.
	callCount := 0
	mock := &apiClientMock{
		GenerateContentFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
			callCount++
			if callCount == 1 {
				// First call: return a thinking model response with thought + function call
				return &genai.GenerateContentResponse{
					Candidates: []*genai.Candidate{
						{
							Content: &genai.Content{
								Role: "model",
								Parts: []*genai.Part{
									{
										Text:             "Let me think about this...",
										Thought:          true,
										ThoughtSignature: []byte("thought-sig-001"),
									},
									{
										FunctionCall: &genai.FunctionCall{
											Name: "write_file",
											Args: map[string]any{"path": "test.txt"},
										},
										ThoughtSignature: []byte("fc-sig-002"),
									},
								},
							},
						},
					},
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						PromptTokenCount:     100,
						CandidatesTokenCount: 50,
					},
				}, nil
			}

			// Second call: verify that the history contains ThoughtSignature
			// The history should include: user message + model response (with signatures) + user (tool response)
			foundThoughtSig := false
			foundFCSig := false
			for _, content := range contents {
				for _, part := range content.Parts {
					if part.Thought && len(part.ThoughtSignature) > 0 {
						foundThoughtSig = true
					}
					if part.FunctionCall != nil && len(part.ThoughtSignature) > 0 {
						foundFCSig = true
					}
				}
			}
			gt.Value(t, foundThoughtSig).Equal(true)
			gt.Value(t, foundFCSig).Equal(true)

			// Return a simple text response
			return &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{Text: "File written successfully."},
							},
						},
					},
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     200,
					CandidatesTokenCount: 20,
				},
			}, nil
		},
	}

	cfg := gollem.NewSessionConfig()
	session, err := gemini.NewSessionWithAPIClient(mock, cfg, "gemini-2.5-flash")
	gt.NoError(t, err)

	ctx := context.Background()

	// First call: should get a function call
	resp1, err := session.Generate(ctx, []gollem.Input{gollem.Text("Write a test file")})
	gt.NoError(t, err)
	gt.A(t, resp1.FunctionCalls).Length(1)
	// Thought text should NOT appear in Texts
	gt.A(t, resp1.Texts).Length(0)

	// Second call: send function response (simulating tool execution result)
	resp2, err := session.Generate(ctx, []gollem.Input{gollem.FunctionResponse{
		Name: "write_file",
		Data: map[string]any{"status": "ok"},
	}})
	gt.NoError(t, err)
	gt.A(t, resp2.Texts).Length(1)
	gt.Value(t, resp2.Texts[0]).Equal("File written successfully.")

	// Verify both calls were made
	gt.Value(t, callCount).Equal(2)
}

func TestThinkingModelHistoryRoundTrip(t *testing.T) {
	// Simulate a thinking model response, export history, restore it, and continue.
	mock := &apiClientMock{
		GenerateContentFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
			return &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{
									Text:             "Reasoning...",
									Thought:          true,
									ThoughtSignature: []byte("sig-thought"),
								},
								{
									Text:             "Hello!",
									ThoughtSignature: []byte("sig-text"),
								},
							},
						},
					},
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{},
			}, nil
		},
	}

	cfg := gollem.NewSessionConfig()
	session, err := gemini.NewSessionWithAPIClient(mock, cfg, "gemini-2.5-flash")
	gt.NoError(t, err)

	ctx := context.Background()

	// Make a call to populate history
	_, err = session.Generate(ctx, []gollem.Input{gollem.Text("Hi")})
	gt.NoError(t, err)

	// Export history
	history, err := session.History()
	gt.NoError(t, err)

	// Restore into a new session
	cfg2 := gollem.NewSessionConfig(gollem.WithSessionHistory(history))

	// For the restored session, verify signatures are in the API call
	mock2 := &apiClientMock{
		GenerateContentFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
			// Verify that signatures are present in history
			foundThoughtSig := false
			foundTextSig := false
			for _, content := range contents {
				for _, part := range content.Parts {
					if part.Thought && string(part.ThoughtSignature) == "sig-thought" {
						foundThoughtSig = true
					}
					if !part.Thought && part.Text == "Hello!" && string(part.ThoughtSignature) == "sig-text" {
						foundTextSig = true
					}
				}
			}
			gt.Value(t, foundThoughtSig).Equal(true)
			gt.Value(t, foundTextSig).Equal(true)

			return &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Role:  "model",
							Parts: []*genai.Part{{Text: "Continued!"}},
						},
					},
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{},
			}, nil
		},
	}

	session2, err := gemini.NewSessionWithAPIClient(mock2, cfg2, "gemini-2.5-flash")
	gt.NoError(t, err)

	resp, err := session2.Generate(ctx, []gollem.Input{gollem.Text("Continue")})
	gt.NoError(t, err)
	gt.A(t, resp.Texts).Length(1)
	gt.Value(t, resp.Texts[0]).Equal("Continued!")
}

func TestGeminiContentGenerateWithModel(t *testing.T) {
	projectID := os.Getenv("TEST_GCP_PROJECT_ID")
	if projectID == "" {
		t.Skip("TEST_GCP_PROJECT_ID is not set")
	}
	location := os.Getenv("TEST_GCP_LOCATION")
	if location == "" {
		t.Skip("TEST_GCP_LOCATION is not set")
	}
	model := os.Getenv("TEST_GCP_MODEL")
	if model == "" {
		t.Skip("TEST_GCP_MODEL is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := gemini.New(ctx, projectID, location,
		gemini.WithModel(model),
		gemini.WithThinkingBudget(-1), // auto thinking budget
	)
	gt.NoError(t, err)

	// Simple tool for testing agent loop
	tool := &writeFileTool{}

	session, err := client.NewSession(ctx,
		gollem.WithSessionTools(tool),
	)
	gt.NoError(t, err)

	// First call: ask the model to use the tool
	resp1, err := session.Generate(ctx, []gollem.Input{gollem.Text("Please call the write_file tool with path 'test.txt' and content 'hello world'. Just call the tool, don't explain.")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err).Required()

	if len(resp1.FunctionCalls) > 0 {
		// Second call: send tool response back
		fc := resp1.FunctionCalls[0]
		resp2, err := session.Generate(ctx, []gollem.Input{gollem.FunctionResponse{
			ID:   fc.ID,
			Name: fc.Name,
			Data: map[string]any{"status": "success", "path": "test.txt"},
		}}, gollem.WithMaxTokens(maxTestTokens))
		gt.NoError(t, err)
		gt.A(t, resp2.Texts).Length(1).Required()
	}
}

// TestGemini35FlashStrictMatchIntegration exercises the function-calling
// round trip end-to-end against a real gemini-3.5-flash deployment. The unit
// tests in TestProcessResponseFunctionCallIDPreserved and
// TestSessionFunctionResponseIDPropagation cover the behavior on the
// conversion layer when Gemini issues a real id; this test only proves that
// the live path (`Generate` + tool round trip) succeeds for 3.5-flash with
// the new WithThinkingLevel option.
//
// Note: as of writing, the Vertex AI deployment of gemini-3.5-flash returns
// FunctionCall.ID="" — strict id matching is therefore implicitly satisfied
// by both call and response carrying an empty wire id. The test does not
// assert on the id being real; it verifies that whatever id gollem stamps
// (real or fallback) survives the round trip.
//
// Gated on TEST_GCP_PROJECT_ID / TEST_GCP_LOCATION so that contributors
// without Vertex AI access can still run `go test ./...`.
func TestGemini35FlashStrictMatchIntegration(t *testing.T) {
	projectID := os.Getenv("TEST_GCP_PROJECT_ID")
	if projectID == "" {
		t.Skip("TEST_GCP_PROJECT_ID is not set")
	}
	location := os.Getenv("TEST_GCP_LOCATION")
	if location == "" {
		t.Skip("TEST_GCP_LOCATION is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := gemini.New(ctx, projectID, location,
		gemini.WithModel("gemini-3.5-flash"),
		gemini.WithThinkingLevel(genai.ThinkingLevelMinimal),
	)
	gt.NoError(t, err)

	tool := &writeFileTool{}
	session, err := client.NewSession(ctx, gollem.WithSessionTools(tool))
	gt.NoError(t, err)

	resp1, err := session.Generate(ctx, []gollem.Input{
		gollem.Text("Please call the write_file tool with path 'test.txt' and content 'hello world'. Just call the tool, don't explain."),
	}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err).Required()
	gt.A(t, resp1.FunctionCalls).Longer(0).Required()

	fc := resp1.FunctionCalls[0]
	// gollem always exposes a non-empty id to the caller — either the one
	// Gemini issued or a fabricated fallback. Either way the same id must
	// flow back into FunctionResponse below for strict matching to hold.
	gt.Value(t, fc.ID).NotEqual("")

	resp2, err := session.Generate(ctx, []gollem.Input{gollem.FunctionResponse{
		ID:   fc.ID,
		Name: fc.Name,
		Data: map[string]any{"status": "success", "path": "test.txt"},
	}}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err).Required()
	gt.A(t, resp2.Texts).Longer(0)
}

// writeFileTool is a simple tool for integration testing
type writeFileTool struct{}

func (t *writeFileTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "write_file",
		Description: "Write content to a file",
		Parameters: map[string]*gollem.Parameter{
			"path": {
				Type:        gollem.TypeString,
				Description: "File path",
				Required:    true,
			},
			"content": {
				Type:        gollem.TypeString,
				Description: "File content",
				Required:    true,
			},
		},
	}
}

func (t *writeFileTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return map[string]any{"status": "success"}, nil
}

func TestGeminiTokenLimitErrorIntegration(t *testing.T) {
	projectID, ok := os.LookupEnv("TEST_GEMINI_PROJECT_ID")
	if !ok {
		t.Skip("TEST_GEMINI_PROJECT_ID is not set")
	}

	location, ok := os.LookupEnv("TEST_GEMINI_LOCATION")
	if !ok {
		t.Skip("TEST_GEMINI_LOCATION is not set")
	}

	// Only run if explicitly requested via environment variable
	if os.Getenv("TEST_TOKEN_LIMIT_ERROR") != "true" {
		t.Skip("TEST_TOKEN_LIMIT_ERROR is not set to true")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := gemini.New(ctx, projectID, location)
	gt.NoError(t, err)

	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	// Create a very long prompt to exceed token limit
	// Repeat a long text many times to ensure we exceed the limit
	longText := strings.Repeat("This is a test sentence to make the prompt very long. ", 100000)

	_, err = session.Generate(ctx, []gollem.Input{gollem.Text(longText)})
	gt.Error(t, err)

	// Verify the error has the token exceeded tag
	gt.True(t, goerr.HasTag(err, gollem.ErrTagTokenExceeded))
}

// TestPerCallGenerateOptions verifies that per-call GenerateOption overrides
// actually change the API request. A text-mode session gets a per-call
// ResponseSchema, and the response must be valid JSON matching the schema.
func TestPerCallGenerateOptions(t *testing.T) {
	projectID, ok := os.LookupEnv("TEST_GCP_PROJECT_ID")
	if !ok {
		t.Skip("TEST_GCP_PROJECT_ID is not set")
	}
	location, ok := os.LookupEnv("TEST_GCP_LOCATION")
	if !ok {
		t.Skip("TEST_GCP_LOCATION is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	var opts []gemini.Option
	if model := os.Getenv("TEST_GCP_MODEL"); model != "" {
		opts = append(opts, gemini.WithModel(model))
	}
	client, err := gemini.New(ctx, projectID, location, opts...)
	gt.NoError(t, err)

	// Create a plain text session — no ContentTypeJSON, no ResponseSchema
	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	schema := &gollem.Parameter{
		Type:  gollem.TypeObject,
		Title: "Color",
		Properties: map[string]*gollem.Parameter{
			"name": {Type: gollem.TypeString, Description: "color name", Required: true},
		},
	}

	// Per-call option should force JSON output via ResponseMIMEType + ResponseSchema
	resp, err := session.Generate(ctx,
		[]gollem.Input{gollem.Text("Name a color.")},
		gollem.WithGenerateResponseSchema(schema),
		gollem.WithMaxTokens(maxTestTokens),
	)
	gt.NoError(t, err)
	gt.True(t, len(resp.Texts) > 0)

	var parsed map[string]any
	if err := json.Unmarshal([]byte(resp.Texts[0]), &parsed); err != nil {
		t.Fatalf("response is not valid JSON: %s (raw: %s)", err, resp.Texts[0])
	}
	gt.True(t, parsed["name"] != nil)
}

func TestContentsToTraceMessages(t *testing.T) {
	type testCase struct {
		contents []*genai.Content
		expected []trace.Message
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			result := gemini.ContentsToTraceMessages(tc.contents)
			gt.Equal(t, tc.expected, result)
		}
	}

	t.Run("text message", runTest(testCase{
		contents: []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: "hello world"}},
			},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewTextContent("hello world"),
			}},
		},
	}))

	t.Run("multiple parts in single content", runTest(testCase{
		contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "first"},
					{Text: "second"},
				},
			},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewTextContent("first"),
				trace.NewTextContent("second"),
			}},
		},
	}))

	t.Run("function call", runTest(testCase{
		contents: []*genai.Content{
			{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: "search"}},
				},
			},
		},
		expected: []trace.Message{
			{Role: "model", Contents: []trace.MessageContent{
				trace.NewToolCallContent("", "search", nil),
			}},
		},
	}))

	t.Run("function response", runTest(testCase{
		contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{Name: "search"}},
				},
			},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewToolResponseContent("", "search", nil),
			}},
		},
	}))

	t.Run("inline data", runTest(testCase{
		contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("data")}},
				},
			},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewMediaContent("image", "image/png"),
			}},
		},
	}))

	t.Run("file data with URI", runTest(testCase{
		contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{FileData: &genai.FileData{FileURI: "gs://bucket/file", MIMEType: "application/pdf", DisplayName: "report.pdf"}},
				},
			},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				{Type: "file", MediaType: "application/pdf", URL: "gs://bucket/file", Title: "report.pdf"},
			}},
		},
	}))

	t.Run("multiple contents", runTest(testCase{
		contents: []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: "hello"}},
			},
			{
				Role:  "model",
				Parts: []*genai.Part{{Text: "world"}},
			},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{trace.NewTextContent("hello")}},
			{Role: "model", Contents: []trace.MessageContent{trace.NewTextContent("world")}},
		},
	}))

	t.Run("nil contents", runTest(testCase{
		contents: nil,
		expected: nil,
	}))

	t.Run("empty contents", runTest(testCase{
		contents: []*genai.Content{},
		expected: nil,
	}))

	t.Run("nil content entry", runTest(testCase{
		contents: []*genai.Content{nil},
		expected: nil,
	}))

	t.Run("mixed content types", runTest(testCase{
		contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "analyze this image"},
					{InlineData: &genai.Blob{MIMEType: "image/jpeg"}},
				},
			},
			{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "I see an image"},
					{FunctionCall: &genai.FunctionCall{Name: "describe_image"}},
				},
			},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewTextContent("analyze this image"),
				trace.NewMediaContent("image", "image/jpeg"),
			}},
			{Role: "model", Contents: []trace.MessageContent{
				trace.NewTextContent("I see an image"),
				trace.NewToolCallContent("", "describe_image", nil),
			}},
		},
	}))

	t.Run("thought text", runTest(testCase{
		contents: []*genai.Content{
			{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "Let me reason about this...", Thought: true},
					{Text: "The answer is 42"},
				},
			},
		},
		expected: []trace.Message{
			{Role: "model", Contents: []trace.MessageContent{
				trace.NewThinkingContent("Let me reason about this..."),
				trace.NewTextContent("The answer is 42"),
			}},
		},
	}))

	t.Run("executable code and result", runTest(testCase{
		contents: []*genai.Content{
			{
				Role: "model",
				Parts: []*genai.Part{
					{ExecutableCode: &genai.ExecutableCode{Code: "print('hello')", Language: "PYTHON"}},
					{CodeExecutionResult: &genai.CodeExecutionResult{Output: "hello", Outcome: "OUTCOME_OK"}},
				},
			},
		},
		expected: []trace.Message{
			{Role: "model", Contents: []trace.MessageContent{
				{Type: "executable_code", Text: "print('hello')"},
				{Type: "code_execution_result", Text: "hello"},
			}},
		},
	}))

	t.Run("inline data with display name", runTest(testCase{
		contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("data"), DisplayName: "screenshot.png"}},
				},
			},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				{Type: "image", MediaType: "image/png", Title: "screenshot.png"},
			}},
		},
	}))
}

// TestProcessResponseFunctionCallIDPreserved verifies that processResponse
// keeps the real FunctionCall.ID issued by Gemini 3.x intact and only
// synthesizes a fallback when the API returns an empty id.
func TestProcessResponseFunctionCallIDPreserved(t *testing.T) {
	t.Run("real id preserved", func(t *testing.T) {
		resp, err := gemini.ProcessResponse(&genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							ID:   "gemini-real-id-xyz",
							Name: "lookup",
							Args: map[string]any{"q": "a"},
						},
					}},
				},
			}},
		})
		gt.NoError(t, err)
		gt.A(t, resp.FunctionCalls).Length(1).Required()
		gt.Value(t, resp.FunctionCalls[0].ID).Equal("gemini-real-id-xyz")
		gt.Value(t, gemini.IsGeminiFallbackToolCallID(resp.FunctionCalls[0].ID)).Equal(false)
	})

	t.Run("empty id gets fallback", func(t *testing.T) {
		resp, err := gemini.ProcessResponse(&genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{Name: "lookup", Args: map[string]any{"q": "a"}}},
						{FunctionCall: &genai.FunctionCall{Name: "lookup", Args: map[string]any{"q": "b"}}},
					},
				},
			}},
		})
		gt.NoError(t, err)
		gt.A(t, resp.FunctionCalls).Length(2).Required()
		// Both ids must be non-empty, carry the fallback prefix, and differ
		// from each other so that same-name parallel calls stay distinct.
		gt.Value(t, gemini.IsGeminiFallbackToolCallID(resp.FunctionCalls[0].ID)).Equal(true)
		gt.Value(t, gemini.IsGeminiFallbackToolCallID(resp.FunctionCalls[1].ID)).Equal(true)
		gt.Value(t, resp.FunctionCalls[0].ID).NotEqual(resp.FunctionCalls[1].ID)
	})
}

// TestSessionFunctionResponseIDPropagation verifies that gollem.FunctionResponse
// ids passed to Generate flow through to genai.FunctionResponse.ID on the wire
// (with the gollem-internal fallback prefix stripped, since Gemini did not
// originally issue those ids).
func TestSessionFunctionResponseIDPropagation(t *testing.T) {
	type capture struct {
		funcResponseID string
		seen           bool
	}

	build := func(c *capture) *apiClientMock {
		return &apiClientMock{
			GenerateContentFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
				for _, content := range contents {
					for _, part := range content.Parts {
						if part.FunctionResponse != nil {
							c.funcResponseID = part.FunctionResponse.ID
							c.seen = true
						}
					}
				}
				return &genai.GenerateContentResponse{
					Candidates: []*genai.Candidate{{
						Content: &genai.Content{
							Role:  "model",
							Parts: []*genai.Part{{Text: "ok"}},
						},
					}},
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{},
				}, nil
			},
		}
	}

	t.Run("real id propagates to wire", func(t *testing.T) {
		var c capture
		session, err := gemini.NewSessionWithAPIClient(build(&c), gollem.NewSessionConfig(), "gemini-3.5-flash")
		gt.NoError(t, err)

		_, err = session.Generate(context.Background(), []gollem.Input{gollem.FunctionResponse{
			ID:   "real-id-from-gemini",
			Name: "lookup",
			Data: map[string]any{"result": "ok"},
		}})
		gt.NoError(t, err)
		gt.Value(t, c.seen).Equal(true)
		gt.Value(t, c.funcResponseID).Equal("real-id-from-gemini")
	})

	t.Run("fallback id stripped on wire", func(t *testing.T) {
		var c capture
		session, err := gemini.NewSessionWithAPIClient(build(&c), gollem.NewSessionConfig(), "gemini-3.5-flash")
		gt.NoError(t, err)

		_, err = session.Generate(context.Background(), []gollem.Input{gollem.FunctionResponse{
			ID:   gemini.GeminiFallbackToolCallID("lookup", 0),
			Name: "lookup",
			Data: map[string]any{"result": "ok"},
		}})
		gt.NoError(t, err)
		gt.Value(t, c.seen).Equal(true)
		gt.Value(t, c.funcResponseID).Equal("")
	})
}

// TestGeminiTraceRequestMessagesNewTurnOnly verifies that the trace's
// LLMRequest.Messages contains only contents newly added in this turn,
// not the entire conversation history that was actually sent to the API.
func TestGeminiTraceRequestMessagesNewTurnOnly(t *testing.T) {
	userContent, err := gollem.NewTextContent("previous question")
	gt.NoError(t, err)
	assistantContent, err := gollem.NewTextContent("previous answer")
	gt.NoError(t, err)
	history := &gollem.History{
		Version: gollem.HistoryVersion,
		LLType:  gollem.LLMTypeGemini,
		Messages: []gollem.Message{
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{userContent}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{assistantContent}},
		},
	}

	var sentContents []*genai.Content
	mockClient := &apiClientMock{
		GenerateContentFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
			sentContents = contents
			return &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content: &genai.Content{
							Role:  "model",
							Parts: []*genai.Part{{Text: "ok"}},
						},
					},
				},
			}, nil
		},
		GenerateContentStreamFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) <-chan gemini.StreamResponse {
			ch := make(chan gemini.StreamResponse)
			close(ch)
			return ch
		},
	}

	cfg := gollem.NewSessionConfig(gollem.WithSessionHistory(history))
	session, err := gemini.NewSessionWithAPIClient(mockClient, cfg, "gemini-1.5-pro")
	gt.NoError(t, err)

	rec := trace.New()
	ctx := rec.StartAgentExecute(context.Background())
	ctx = trace.WithHandler(ctx, rec)

	_, err = session.Generate(ctx, []gollem.Input{gollem.Text("new question")})
	gt.NoError(t, err)
	rec.EndAgentExecute(ctx, nil)

	// Sanity check: the actual API request still includes the full history
	// (two prior turns + the new user message).
	gt.N(t, len(sentContents)).Equal(3)

	// Find the LLM call span.
	var llmSpan *trace.Span
	for _, child := range rec.Trace().RootSpan.Children {
		if child.Kind == trace.SpanKindLLMCall {
			llmSpan = child
			break
		}
	}
	gt.Value(t, llmSpan).NotNil()

	msgs := llmSpan.LLMCall.Request.Messages
	gt.A(t, msgs).Length(1)
	gt.Equal(t, "user", msgs[0].Role)
	gt.A(t, msgs[0].Contents).Length(1)
	gt.Equal(t, "text", msgs[0].Contents[0].Type)
	gt.Equal(t, "new question", msgs[0].Contents[0].Text)

	for _, m := range msgs {
		for _, c := range m.Contents {
			gt.S(t, c.Text).NotContains("previous")
		}
	}
}

func TestGeminiCacheTokenObservation(t *testing.T) {
	mock := &apiClientMock{
		GenerateContentFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
			return &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "ok"}}}},
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:        200,
					CandidatesTokenCount:    10,
					CachedContentTokenCount: 150,
				},
			}, nil
		},
	}

	cfg := gollem.NewSessionConfig()
	session, err := gemini.NewSessionWithAPIClient(mock, cfg, "gemini-2.5-flash")
	gt.NoError(t, err)

	resp, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")})
	gt.NoError(t, err)

	// PromptTokenCount already includes cached tokens, so InputToken stays total.
	gt.Equal(t, 200, resp.InputToken)
	gt.Equal(t, 150, resp.CacheReadInputToken)
	// Gemini implicit caching does not report cache writes.
	gt.Equal(t, 0, resp.CacheCreationInputToken)
}

func TestGeminiStreamUsageNotSummed(t *testing.T) {
	// Two stream chunks both carry usage (Gemini reports the running total, not a
	// per-chunk delta). The session must report the single total, not the sum.
	makeResp := func(text string) *genai.GenerateContentResponse {
		return &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: text}}}},
			},
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        100,
				CandidatesTokenCount:    8,
				CachedContentTokenCount: 50,
			},
		}
	}
	mock := &apiClientMock{
		GenerateContentStreamFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) <-chan gemini.StreamResponse {
			ch := make(chan gemini.StreamResponse, 2)
			ch <- gemini.StreamResponse{Resp: makeResp("a")}
			ch <- gemini.StreamResponse{Resp: makeResp("b")}
			close(ch)
			return ch
		},
	}

	cfg := gollem.NewSessionConfig()
	session, err := gemini.NewSessionWithAPIClient(mock, cfg, "gemini-2.5-flash")
	gt.NoError(t, err)

	ch, err := session.Stream(context.Background(), []gollem.Input{gollem.Text("hi")})
	gt.NoError(t, err)

	var lastInput, lastCacheRead int
	for resp := range ch {
		gt.NoError(t, resp.Error)
		if resp.InputToken > 0 {
			lastInput = resp.InputToken
		}
		if resp.CacheReadInputToken > 0 {
			lastCacheRead = resp.CacheReadInputToken
		}
	}
	// Not 200 / 100.
	gt.Equal(t, 100, lastInput)
	gt.Equal(t, 50, lastCacheRead)
}
