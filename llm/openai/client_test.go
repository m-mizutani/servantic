package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
	"github.com/m-mizutani/gt"
	openaiapi "github.com/sashabaranov/go-openai"
)

const (
	testTimeout   = 30 * time.Second
	maxTestTokens = 2048
)

func TestOpenAIContentGenerate(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := openai.New(ctx, apiKey)
	gt.NoError(t, err)

	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	result, err := session.Generate(ctx, []gollem.Input{gollem.Text("Say hello in one word")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err)
	gt.Array(t, result.Texts).Length(1).Required()
	gt.Value(t, len(result.Texts[0])).NotEqual(0)
}

func TestTokenLimitErrorOptions(t *testing.T) {
	type testCase struct {
		name   string
		err    error
		hasTag bool
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			opts := openai.TokenLimitErrorOptions(tc.err)
			if tc.hasTag {
				gt.NotEqual(t, 0, len(opts))
			} else {
				gt.Equal(t, 0, len(opts))
			}
		}
	}

	t.Run("token exceeded error", runTest(testCase{
		name: "context_length_exceeded",
		err: &openaiapi.APIError{
			Type:    "invalid_request_error",
			Code:    "context_length_exceeded",
			Message: "This model's maximum context length is 128000 tokens. However, your messages resulted in 150000 tokens.",
		},
		hasTag: true,
	}))

	t.Run("different error type", runTest(testCase{
		name: "different type",
		err: &openaiapi.APIError{
			Type:    "authentication_error",
			Code:    "invalid_api_key",
			Message: "Invalid API key",
		},
		hasTag: false,
	}))

	t.Run("different error code", runTest(testCase{
		name: "different code",
		err: &openaiapi.APIError{
			Type:    "invalid_request_error",
			Code:    "invalid_model",
			Message: "The model does not exist",
		},
		hasTag: false,
	}))

	t.Run("code is not string", runTest(testCase{
		name: "code as int",
		err: &openaiapi.APIError{
			Type:    "invalid_request_error",
			Code:    12345,
			Message: "Some error",
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

func TestOpenAITokenLimitErrorIntegration(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}

	// Only run if explicitly requested via environment variable
	if os.Getenv("TEST_TOKEN_LIMIT_ERROR") != "true" {
		t.Skip("TEST_TOKEN_LIMIT_ERROR is not set to true")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	// Use gpt-5 (default model) which has 128k context limit
	client, err := openai.New(ctx, apiKey)
	gt.NoError(t, err)

	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	// Create a very long prompt to exceed token limit
	// gpt-5 may have larger context limit, so we create much more text
	// Approximately 1 token = 4 characters, aim for ~300k+ tokens
	longText := strings.Repeat("This is a test sentence to make the prompt very long. ", 25000)

	_, err = session.Generate(ctx, []gollem.Input{gollem.Text(longText)})
	gt.Error(t, err)

	// Log error details for debugging
	t.Logf("Error: %+v", err)
	t.Logf("Error tags: %v", goerr.Tags(err))

	// Verify the error has the token exceeded tag
	gt.True(t, goerr.HasTag(err, gollem.ErrTagTokenExceeded))
}

// TestPerCallGenerateOptions verifies that per-call GenerateOption overrides
// actually change the API request. A text-mode session gets a per-call
// ResponseSchema, and the response must be valid JSON matching the schema.
func TestPerCallGenerateOptions(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := openai.New(ctx, apiKey)
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

	// Per-call option should force JSON schema output
	resp, err := session.Generate(ctx,
		[]gollem.Input{gollem.Text("Name a color.")},
		gollem.WithGenerateResponseSchema(schema),
		gollem.WithMaxTokens(maxTestTokens),
	)
	gt.NoError(t, err)
	gt.True(t, len(resp.Texts) > 0)

	// The response must be valid JSON
	var parsed map[string]any
	gt.NoError(t, json.Unmarshal([]byte(resp.Texts[0]), &parsed))
	gt.True(t, parsed["name"] != nil)
}

// TestWithBaseURL tests the WithBaseURL option functionality for OpenAI
// Reference: Brain Memory c4705651-435d-4cca-95eb-d39d1ea69a9c
func TestWithBaseURL(t *testing.T) {
	t.Run("default baseURL", func(t *testing.T) {
		client, err := openai.New(context.Background(), "test-key", openai.WithBaseURL(""))
		gt.NoError(t, err)
		gt.Equal(t, "", openai.GetBaseURL(client))
	})

	t.Run("custom baseURL", func(t *testing.T) {
		customURL := "https://api.custom-openai.com"
		client, err := openai.New(context.Background(), "test-key", openai.WithBaseURL(customURL))
		gt.NoError(t, err)
		gt.Equal(t, customURL, openai.GetBaseURL(client))
	})

	t.Run("empty baseURL after custom", func(t *testing.T) {
		// Test that empty baseURL overrides previous setting
		client1, err1 := openai.New(context.Background(), "test-key", openai.WithBaseURL("https://first.com"))
		gt.NoError(t, err1)
		gt.Equal(t, "https://first.com", openai.GetBaseURL(client1))

		// Apply empty baseURL after custom one
		client2, err2 := openai.New(context.Background(), "test-key",
			openai.WithBaseURL("https://first.com"),
			openai.WithBaseURL(""))
		gt.NoError(t, err2)
		gt.Equal(t, "", openai.GetBaseURL(client2)) // Should be empty, not first URL
	})
}

func TestOpenaiMessagesToTraceMessages(t *testing.T) {
	type testCase struct {
		messages []openaiapi.ChatCompletionMessage
		expected []trace.Message
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			result := openai.OpenaiMessagesToTraceMessages(tc.messages)
			gt.Equal(t, tc.expected, result)
		}
	}

	t.Run("user text message", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{Role: openaiapi.ChatMessageRoleUser, Content: "hello world"},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewTextContent("hello world"),
			}},
		},
	}))

	t.Run("system message", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{Role: openaiapi.ChatMessageRoleSystem, Content: "you are helpful"},
		},
		expected: []trace.Message{
			{Role: "system", Contents: []trace.MessageContent{
				trace.NewTextContent("you are helpful"),
			}},
		},
	}))

	t.Run("assistant with tool calls", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{
				Role: openaiapi.ChatMessageRoleAssistant,
				ToolCalls: []openaiapi.ToolCall{
					{
						ID:   "call-1",
						Type: openaiapi.ToolTypeFunction,
						Function: openaiapi.FunctionCall{
							Name:      "search",
							Arguments: `{"q":"test"}`,
						},
					},
				},
			},
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewToolCallContent("call-1", "search", map[string]any{"q": "test"}),
			}},
		},
	}))

	t.Run("tool response", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{
				Role:       openaiapi.ChatMessageRoleTool,
				Content:    "search result",
				ToolCallID: "call-1",
			},
		},
		expected: []trace.Message{
			{Role: "tool", Contents: []trace.MessageContent{
				{Type: "tool_response", ToolCallID: "call-1", Text: "search result"},
			}},
		},
	}))

	t.Run("multi content with image URL", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{
				Role: openaiapi.ChatMessageRoleUser,
				MultiContent: []openaiapi.ChatMessagePart{
					{Type: openaiapi.ChatMessagePartTypeText, Text: "describe this"},
					{Type: openaiapi.ChatMessagePartTypeImageURL, ImageURL: &openaiapi.ChatMessageImageURL{URL: "https://example.com/img.png"}},
				},
			},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewTextContent("describe this"),
				{Type: "image", URL: "https://example.com/img.png"},
			}},
		},
	}))

	t.Run("assistant text with tool calls", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{
				Role:    openaiapi.ChatMessageRoleAssistant,
				Content: "Let me search",
				ToolCalls: []openaiapi.ToolCall{
					{
						ID:   "call-1",
						Type: openaiapi.ToolTypeFunction,
						Function: openaiapi.FunctionCall{
							Name:      "search",
							Arguments: `{"q":"test"}`,
						},
					},
				},
			},
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewTextContent("Let me search"),
				trace.NewToolCallContent("call-1", "search", map[string]any{"q": "test"}),
			}},
		},
	}))

	t.Run("multiple messages", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{Role: openaiapi.ChatMessageRoleUser, Content: "hello"},
			{Role: openaiapi.ChatMessageRoleAssistant, Content: "hi"},
			{Role: openaiapi.ChatMessageRoleUser, Content: "how are you"},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{trace.NewTextContent("hello")}},
			{Role: "assistant", Contents: []trace.MessageContent{trace.NewTextContent("hi")}},
			{Role: "user", Contents: []trace.MessageContent{trace.NewTextContent("how are you")}},
		},
	}))

	t.Run("nil messages", runTest(testCase{
		messages: nil,
		expected: nil,
	}))

	t.Run("empty messages", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{},
		expected: nil,
	}))
}

func TestThinkingContentExtraction(t *testing.T) {
	t.Run("non-streaming response with reasoning content", func(t *testing.T) {
		// Create a mock API client that returns reasoning content
		mockClient := &apiClientMock{
			CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
				return openaiapi.ChatCompletionResponse{
					Model: "gpt-5",
					Choices: []openaiapi.ChatCompletionChoice{
						{
							Message: openaiapi.ChatCompletionMessage{
								Role:             openaiapi.ChatMessageRoleAssistant,
								Content:          "This is the final answer",
								ReasoningContent: "Let me think through this step by step...",
							},
							FinishReason: openaiapi.FinishReasonStop,
						},
					},
					Usage: openaiapi.Usage{
						PromptTokens:     10,
						CompletionTokens: 20,
					},
				}, nil
			},
		}

		cfg := gollem.NewSessionConfig()
		session, err := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-5")
		gt.NoError(t, err)

		result, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("Test input")})
		gt.NoError(t, err)
		gt.Equal(t, []string{"This is the final answer"}, result.Texts)
		gt.Equal(t, []string{"Let me think through this step by step..."}, result.Thoughts)
		gt.Equal(t, 10, result.InputToken)
		gt.Equal(t, 20, result.OutputToken)
	})

	t.Run("non-streaming response without reasoning content", func(t *testing.T) {
		// Create a mock API client that returns only text content
		mockClient := &apiClientMock{
			CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
				return openaiapi.ChatCompletionResponse{
					Model: "gpt-5",
					Choices: []openaiapi.ChatCompletionChoice{
						{
							Message: openaiapi.ChatCompletionMessage{
								Role:    openaiapi.ChatMessageRoleAssistant,
								Content: "This is the answer",
							},
							FinishReason: openaiapi.FinishReasonStop,
						},
					},
					Usage: openaiapi.Usage{
						PromptTokens:     10,
						CompletionTokens: 15,
					},
				}, nil
			},
		}

		cfg := gollem.NewSessionConfig()
		session, err := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-5")
		gt.NoError(t, err)

		result, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("Test input")})
		gt.NoError(t, err)
		gt.Equal(t, []string{"This is the answer"}, result.Texts)
		gt.Equal(t, []string{}, result.Thoughts) // Should be empty slice
		gt.Equal(t, 10, result.InputToken)
		gt.Equal(t, 15, result.OutputToken)
	})
}

// TestOpenAITraceRequestMessagesNewTurnOnly verifies that the trace's
// LLMRequest.Messages contains only messages newly added in this turn,
// not the entire conversation history that was actually sent to the API.
func TestOpenAITraceRequestMessagesNewTurnOnly(t *testing.T) {
	makeHistory := func() *gollem.History {
		userContent, err := gollem.NewTextContent("previous question")
		gt.NoError(t, err)
		assistantContent, err := gollem.NewTextContent("previous answer")
		gt.NoError(t, err)
		return &gollem.History{
			Version: gollem.HistoryVersion,
			LLType:  gollem.LLMTypeOpenAI,
			Messages: []gollem.Message{
				{Role: gollem.RoleUser, Contents: []gollem.MessageContent{userContent}},
				{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{assistantContent}},
			},
		}
	}

	findLLMSpan := func(t *testing.T, root *trace.Span) *trace.Span {
		t.Helper()
		for _, child := range root.Children {
			if child.Kind == trace.SpanKindLLMCall {
				return child
			}
		}
		t.Fatal("llm_call span not found")
		return nil
	}

	t.Run("Generate", func(t *testing.T) {
		var sentMessages []openaiapi.ChatCompletionMessage
		mockClient := &apiClientMock{
			CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
				sentMessages = req.Messages
				return openaiapi.ChatCompletionResponse{
					Model: "gpt-5",
					Choices: []openaiapi.ChatCompletionChoice{
						{
							Message: openaiapi.ChatCompletionMessage{
								Role:    openaiapi.ChatMessageRoleAssistant,
								Content: "ok",
							},
							FinishReason: openaiapi.FinishReasonStop,
						},
					},
				}, nil
			},
		}

		cfg := gollem.NewSessionConfig(gollem.WithSessionHistory(makeHistory()))
		session, err := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-5")
		gt.NoError(t, err)

		rec := trace.New()
		ctx := rec.StartAgentExecute(context.Background())
		ctx = trace.WithHandler(ctx, rec)

		_, err = session.Generate(ctx, []gollem.Input{gollem.Text("new question")})
		gt.NoError(t, err)
		rec.EndAgentExecute(ctx, nil)

		// Sanity check: the actual API request includes the full history.
		gt.N(t, len(sentMessages)).Equal(3)

		// But the trace records only the new turn.
		llmSpan := findLLMSpan(t, rec.Trace().RootSpan)
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
	})

	// Note: Stream is not covered by a mock-based test because
	// *openai.ChatCompletionStream cannot be constructed outside the SDK
	// (it has unexported fields). The Stream code path uses the same
	// newMessages variable and the same openaiMessagesToTraceMessages call
	// site as Generate, so the Generate test above is structurally
	// equivalent for the trace-delta invariant.
}

func TestOpenAICacheTokenObservation(t *testing.T) {
	runTest := func(details *openaiapi.PromptTokensDetails, wantCacheRead int) func(t *testing.T) {
		return func(t *testing.T) {
			mockClient := &apiClientMock{
				CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
					return openaiapi.ChatCompletionResponse{
						Choices: []openaiapi.ChatCompletionChoice{
							{Message: openaiapi.ChatCompletionMessage{Content: "ok", Role: openaiapi.ChatMessageRoleAssistant}},
						},
						Usage: openaiapi.Usage{
							PromptTokens:        200,
							CompletionTokens:    10,
							PromptTokensDetails: details,
						},
					}, nil
				},
			}

			cfg := gollem.NewSessionConfig()
			session, err := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-4")
			gt.NoError(t, err)

			resp, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")})
			gt.NoError(t, err)

			// OpenAI's PromptTokens already includes cached tokens, so InputToken stays total.
			gt.Equal(t, 200, resp.InputToken)
			gt.Equal(t, wantCacheRead, resp.CacheReadInputToken)
			// OpenAI does not report cache writes.
			gt.Equal(t, 0, resp.CacheCreationInputToken)
		}
	}

	t.Run("reports cached prompt tokens", runTest(&openaiapi.PromptTokensDetails{CachedTokens: 150}, 150))
	t.Run("nil details yields zero", runTest(nil, 0))
}

// TestOpenAIStreamUsageLive verifies that streaming reports token usage. Before
// the fix the loop broke on the finish reason and never read the trailing usage
// chunk, so streamed responses reported zero tokens.
func TestOpenAIStreamUsageLive(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := openai.New(ctx, apiKey)
	gt.NoError(t, err).Required()
	session, err := client.NewSession(ctx)
	gt.NoError(t, err).Required()

	ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("Say hello in one word")}, gollem.WithMaxTokens(64))
	gt.NoError(t, err).Required()

	var lastInput, lastOutput int
	for resp := range ch {
		gt.NoError(t, resp.Error)
		if resp.InputToken > 0 {
			lastInput = resp.InputToken
		}
		if resp.OutputToken > 0 {
			lastOutput = resp.OutputToken
		}
	}
	t.Logf("stream usage: input=%d output=%d", lastInput, lastOutput)
	gt.Value(t, lastInput > 0).Equal(true)
	gt.Value(t, lastOutput > 0).Equal(true)
}

// TestConvertResponseSchemaToOpenAIIsByteStable pins the response schema JSON to
// be byte-identical between conversions. In strict mode every property name is
// copied into the required array, which is the array most exposed to Go map
// iteration order.
func TestConvertResponseSchemaToOpenAIIsByteStable(t *testing.T) {
	param := &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"zulu":  {Type: gollem.TypeString, Required: true},
			"alpha": {Type: gollem.TypeString, Required: true},
			"mike":  {Type: gollem.TypeString},
			"bravo": {Type: gollem.TypeString},
		},
	}

	runTest := func(strict bool, expectedRequired string) func(t *testing.T) {
		return func(t *testing.T) {
			first, err := openai.ConvertResponseSchemaToOpenAI(param, strict)
			gt.NoError(t, err)
			gt.S(t, string(first.Schema.(json.RawMessage))).Contains(expectedRequired)

			for i := 0; i < 100; i++ {
				actual, err := openai.ConvertResponseSchemaToOpenAI(param, strict)
				gt.NoError(t, err)
				gt.Equal(t,
					string(first.Schema.(json.RawMessage)),
					string(actual.Schema.(json.RawMessage)))
			}
		}
	}

	t.Run("strict mode requires every property", runTest(true, `"required":["alpha","bravo","mike","zulu"]`))
	t.Run("non-strict mode requires the marked properties", runTest(false, `"required":["alpha","zulu"]`))
}
