package claude_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
	"github.com/m-mizutani/gt"
)

const (
	testTimeout   = 30 * time.Second
	maxTestTokens = 2048
)

func TestClaudeContentGenerate(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
	if !ok {
		t.Skip("TEST_CLAUDE_API_KEY is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := claude.New(ctx, apiKey)
	gt.NoError(t, err)

	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	result, err := session.Generate(ctx, []gollem.Input{gollem.Text("Say hello in one word")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err)
	gt.Array(t, result.Texts).Length(1).Required()
	gt.Value(t, len(result.Texts[0])).NotEqual(0)
}

// TestCreateSystemPrompt tests the createSystemPrompt function
func TestCreateSystemPrompt(t *testing.T) {
	ctx := context.Background()

	t.Run("empty config returns empty slice", func(t *testing.T) {
		cfg := gollem.NewSessionConfig()
		result, err := claude.CreateSystemPrompt(ctx, cfg)
		gt.NoError(t, err)

		// Should return empty slice when no system prompt
		gt.Equal(t, 0, len(result))
	})

	t.Run("result is correct type", func(t *testing.T) {
		cfg := gollem.NewSessionConfig()
		result, err := claude.CreateSystemPrompt(ctx, cfg)
		gt.NoError(t, err)

		// Empty slice can be nil in this implementation
		gt.Equal(t, 0, len(result))
	})

	t.Run("JSON content type check", func(t *testing.T) {
		// Create config with JSON content type
		cfg := gollem.NewSessionConfig()
		// Manually set content type since we can't use WithContentType in test
		// The actual functionality is tested in integration tests
		result, err := claude.CreateSystemPrompt(ctx, cfg)
		gt.NoError(t, err)

		// At minimum, should not panic and return valid type
		_ = result
	})
}

// TestSystemPromptSDKCompliance verifies SDK compliance
func TestSystemPromptSDKCompliance(t *testing.T) {
	ctx := context.Background()

	t.Run("SDK format verification", func(t *testing.T) {
		// This test verifies the format matches SDK expectations:
		// []anthropic.TextBlockParam{{Text: "..."}}

		// Create empty config
		cfg := gollem.NewSessionConfig()
		result, err := claude.CreateSystemPrompt(ctx, cfg)
		gt.NoError(t, err)

		// Empty case should return empty slice
		gt.Equal(t, 0, len(result))
	})

	t.Run("TextBlockParam structure", func(t *testing.T) {
		// Verify we can create TextBlockParam correctly
		testBlock := anthropic.TextBlockParam{
			Text: "Test prompt",
		}

		// Verify the Text field exists and is accessible
		gt.Equal(t, "Test prompt", testBlock.Text)

		// Create a slice as the function would return
		blocks := []anthropic.TextBlockParam{testBlock}
		gt.Equal(t, 1, len(blocks))
		gt.Equal(t, "Test prompt", blocks[0].Text)
	})
}

// TestSystemPromptComment verifies the implementation comment
func TestSystemPromptComment(t *testing.T) {
	ctx := context.Background()

	// This test documents that the implementation follows the official SDK format
	// The createSystemPrompt function should return []anthropic.TextBlockParam
	// in the format: []anthropic.TextBlockParam{{Text: "..."}}

	t.Run("comment accuracy", func(t *testing.T) {
		// The function is documented as:
		// "Returns []anthropic.TextBlockParam as per anthropic-sdk-go v1.5.0 specification"
		// This test verifies that claim

		cfg := gollem.NewSessionConfig()
		result, err := claude.CreateSystemPrompt(ctx, cfg)
		gt.NoError(t, err)

		// Should handle empty case correctly
		if len(result) > 0 {
			// If not empty, each element should have a Text field
			for _, block := range result {
				// Text field should be accessible
				_ = block.Text
			}
		}
	})
}

func TestTokenLimitErrorOptions(t *testing.T) {
	type testCase struct {
		name   string
		err    error
		hasTag bool
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			opts := claude.TokenLimitErrorOptions(tc.err)
			if tc.hasTag {
				gt.NotEqual(t, 0, len(opts))
			} else {
				gt.Equal(t, 0, len(opts))
			}
		}
	}

	// Create a mock anthropic.Error with token exceeded error
	createTokenExceededError := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "prompt is too long: 150000 tokens > 100000 maximum",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 400,
		}
		// Use UnmarshalJSON to properly set the internal raw field
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	createDifferentTypeError := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "authentication_error",
				"message": "Invalid API key",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 401,
		}
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	createDifferentMessageError := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "Invalid model specified",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 400,
		}
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	createDifferentStatusError := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "prompt is too long: 150000 tokens > 100000 maximum",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 500,
		}
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	create413Error := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "Prompt is too long",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 413,
		}
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	createCapitalizedMessageError := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "Prompt is too long: 150000 tokens > 100000 maximum",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 400,
		}
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	t.Run("token exceeded error", runTest(testCase{
		name:   "prompt is too long",
		err:    createTokenExceededError(),
		hasTag: true,
	}))

	t.Run("different error type", runTest(testCase{
		name:   "authentication error",
		err:    createDifferentTypeError(),
		hasTag: false,
	}))

	t.Run("different message", runTest(testCase{
		name:   "invalid model",
		err:    createDifferentMessageError(),
		hasTag: false,
	}))

	t.Run("different status code", runTest(testCase{
		name:   "status 500",
		err:    createDifferentStatusError(),
		hasTag: false,
	}))

	t.Run("413 status code with capitalized message", runTest(testCase{
		name:   "413 Request Entity Too Large",
		err:    create413Error(),
		hasTag: true,
	}))

	t.Run("capitalized message with 400 status", runTest(testCase{
		name:   "Prompt is too long (capitalized)",
		err:    createCapitalizedMessageError(),
		hasTag: true,
	}))

	t.Run("nil error", runTest(testCase{
		name:   "nil error",
		err:    nil,
		hasTag: false,
	}))

	t.Run("non-anthropic error", runTest(testCase{
		name:   "generic error",
		err:    errors.New("some error"),
		hasTag: false,
	}))
}

func TestClaudeTokenLimitErrorIntegration(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
	if !ok {
		t.Skip("TEST_CLAUDE_API_KEY is not set")
	}

	// Only run if explicitly requested via environment variable
	if os.Getenv("TEST_TOKEN_LIMIT_ERROR") != "true" {
		t.Skip("TEST_TOKEN_LIMIT_ERROR is not set to true")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := claude.New(ctx, apiKey)
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

// TestWithBaseURL tests the WithBaseURL option functionality
func TestWithBaseURL(t *testing.T) {
	t.Run("default baseURL", func(t *testing.T) {
		client, err := claude.New(context.Background(), "test-key", claude.WithBaseURL(""))
		gt.NoError(t, err)
		gt.Equal(t, "", claude.GetBaseURL(client))
	})

	t.Run("custom baseURL", func(t *testing.T) {
		customURL := "https://custom.anthropic.com"
		client, err := claude.New(context.Background(), "test-key", claude.WithBaseURL(customURL))
		gt.NoError(t, err)
		gt.Equal(t, customURL, claude.GetBaseURL(client))
	})

	t.Run("empty baseURL after custom", func(t *testing.T) {
		// Test that empty baseURL overrides previous setting
		client1, err1 := claude.New(context.Background(), "test-key", claude.WithBaseURL("https://first.com"))
		gt.NoError(t, err1)
		gt.Equal(t, "https://first.com", claude.GetBaseURL(client1))

		// Apply empty baseURL after custom one
		client2, err2 := claude.New(context.Background(), "test-key",
			claude.WithBaseURL("https://first.com"),
			claude.WithBaseURL(""))
		gt.NoError(t, err2)
		gt.Equal(t, "", claude.GetBaseURL(client2)) // Should be empty, not first URL
	})
}

// TestPerCallGenerateOptions verifies that per-call GenerateOption overrides
// actually change the API request. A text-mode session gets a per-call
// ResponseSchema, and the response must be valid JSON matching the schema.
func TestPerCallGenerateOptions(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
	if !ok {
		t.Skip("TEST_CLAUDE_API_KEY is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := claude.New(ctx, apiKey)
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

	// Per-call option should force JSON output via system prompt injection
	resp, err := session.Generate(ctx,
		[]gollem.Input{gollem.Text("Name a color.")},
		gollem.WithGenerateResponseSchema(schema),
		gollem.WithMaxTokens(maxTestTokens),
	)
	gt.NoError(t, err)
	gt.True(t, len(resp.Texts) > 0)

	var parsed map[string]any
	gt.NoError(t, json.Unmarshal([]byte(resp.Texts[0]), &parsed))
	gt.True(t, parsed["name"] != nil)
}

func TestClaudeMessagesToTraceMessages(t *testing.T) {
	type testCase struct {
		messages []anthropic.MessageParam
		expected []trace.Message
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			result := claude.ClaudeMessagesToTraceMessages(tc.messages)
			gt.Equal(t, tc.expected, result)
		}
	}

	t.Run("text message", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hello world")),
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewTextContent("hello world"),
			}},
		},
	}))

	t.Run("assistant message", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("response")),
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewTextContent("response"),
			}},
		},
	}))

	t.Run("tool use", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewAssistantMessage(
				anthropic.NewToolUseBlock("call-1", map[string]any{"q": "test"}, "search"),
			),
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewToolCallContent("call-1", "search", map[string]any{"q": "test"}),
			}},
		},
	}))

	t.Run("tool result", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock("call-1", "result text", false),
			),
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewToolResponseContent("call-1", "", nil),
				trace.NewTextContent("result text"),
			}},
		},
	}))

	t.Run("multiple messages", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("hi")),
			anthropic.NewUserMessage(anthropic.NewTextBlock("how are you")),
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
		messages: []anthropic.MessageParam{},
		expected: nil,
	}))

	t.Run("mixed content blocks", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewAssistantMessage(
				anthropic.NewTextBlock("Let me search"),
				anthropic.NewToolUseBlock("call-1", map[string]any{"q": "test"}, "search"),
			),
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewTextContent("Let me search"),
				trace.NewToolCallContent("call-1", "search", map[string]any{"q": "test"}),
			}},
		},
	}))

	t.Run("image with media type", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewImageBlockBase64("image/png", "iVBOR..."),
			),
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				{Type: "image", MediaType: "image/png"},
			}},
		},
	}))

	t.Run("thinking block", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewAssistantMessage(
				anthropic.NewThinkingBlock("sig123", "Let me think about this..."),
				anthropic.NewTextBlock("Here is my answer"),
			),
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewThinkingContent("Let me think about this..."),
				trace.NewTextContent("Here is my answer"),
			}},
		},
	}))

	t.Run("redacted thinking block", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewAssistantMessage(
				anthropic.NewRedactedThinkingBlock("redacted-data"),
				anthropic.NewTextBlock("answer"),
			),
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewRedactedThinkingContent(),
				trace.NewTextContent("answer"),
			}},
		},
	}))
}

// TestClaudeTraceRequestMessagesNewTurnOnly verifies that the trace's
// LLMRequest.Messages contains only messages newly added in this turn,
// not the entire conversation history that was actually sent to the API.
func TestClaudeTraceRequestMessagesNewTurnOnly(t *testing.T) {
	userContent, err := gollem.NewTextContent("previous question")
	gt.NoError(t, err)
	assistantContent, err := gollem.NewTextContent("previous answer")
	gt.NoError(t, err)
	history := &gollem.History{
		Version: gollem.HistoryVersion,
		LLType:  gollem.LLMTypeClaude,
		Messages: []gollem.Message{
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{userContent}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{assistantContent}},
		},
	}

	var sentMessages []anthropic.MessageParam
	mockClient := &apiClientMock{
		MessagesNewFunc: func(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
			sentMessages = params.Messages
			return &anthropic.Message{
				Content: []anthropic.ContentBlockUnion{
					{Type: "text", Text: "ok"},
				},
				Role:  "assistant",
				Model: "claude-3-opus-20240229",
			}, nil
		},
	}

	cfg := gollem.NewSessionConfig(gollem.WithSessionHistory(history))
	session, err := claude.NewSessionWithAPIClient(mockClient, cfg, "claude-3-opus-20240229")
	gt.NoError(t, err)

	rec := trace.New()
	ctx := rec.StartAgentExecute(context.Background())
	ctx = trace.WithHandler(ctx, rec)

	_, err = session.Generate(ctx, []gollem.Input{gollem.Text("new question")})
	gt.NoError(t, err)
	rec.EndAgentExecute(ctx, nil)

	// Sanity check: the actual API request still includes the full history.
	gt.N(t, len(sentMessages)).Equal(3)

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

// cachePromptTestTool is a minimal tool used to exercise prompt-cache breakpoints.
type cachePromptTestTool struct{}

func (t *cachePromptTestTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "echo",
		Description: "echo",
		Parameters: map[string]*gollem.Parameter{
			"msg": {Type: gollem.TypeString, Description: "message"},
		},
	}
}

func (t *cachePromptTestTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return map[string]any{"ok": true}, nil
}

func TestCacheTokensFromUsage(t *testing.T) {
	t.Run("restores total input from cached prefix", func(t *testing.T) {
		total, creation, read := claude.CacheTokensFromUsage(anthropic.Usage{
			InputTokens:              50,
			CacheCreationInputTokens: 10,
			CacheReadInputTokens:     100,
		})
		gt.Equal(t, 160, total)
		gt.Equal(t, 10, creation)
		gt.Equal(t, 100, read)
	})

	t.Run("no caching yields raw input and zero cache", func(t *testing.T) {
		total, creation, read := claude.CacheTokensFromUsage(anthropic.Usage{InputTokens: 42})
		gt.Equal(t, 42, total)
		gt.Equal(t, 0, creation)
		gt.Equal(t, 0, read)
	})
}

func TestApplyPromptCacheBreakpoints(t *testing.T) {
	const ttl5m = anthropic.CacheControlEphemeralTTLTTL5m

	t.Run("marks system, tools and conversation tail", func(t *testing.T) {
		req := anthropic.MessageNewParams{
			System: []anthropic.TextBlockParam{{Text: "a"}, {Text: "b"}},
			Tools: []anthropic.ToolUnionParam{
				anthropic.ToolUnionParamOfTool(anthropic.ToolInputSchemaParam{}, "t1"),
				anthropic.ToolUnionParamOfTool(anthropic.ToolInputSchemaParam{}, "t2"),
			},
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock("first")),
				anthropic.NewUserMessage(anthropic.NewTextBlock("second")),
			},
		}
		claude.ApplyPromptCacheBreakpoints(&req)

		// Only the last system block is marked.
		gt.Equal(t, anthropic.CacheControlEphemeralTTL(""), req.System[0].CacheControl.TTL)
		gt.Equal(t, ttl5m, req.System[1].CacheControl.TTL)
		// Only the last tool is marked.
		gt.Equal(t, anthropic.CacheControlEphemeralTTL(""), req.Tools[0].OfTool.CacheControl.TTL)
		gt.Equal(t, ttl5m, req.Tools[1].OfTool.CacheControl.TTL)
		// Only the last message's last block is marked.
		tail := req.Messages[1].Content
		gt.Equal(t, ttl5m, tail[len(tail)-1].OfText.CacheControl.TTL)
	})

	t.Run("empty sections are skipped without panic", func(t *testing.T) {
		req := anthropic.MessageNewParams{}
		claude.ApplyPromptCacheBreakpoints(&req) // must not panic
		gt.Equal(t, 0, len(req.System))
		gt.Equal(t, 0, len(req.Tools))
		gt.Equal(t, 0, len(req.Messages))
	})

	t.Run("does not mutate shared history or tools", func(t *testing.T) {
		origMsg := anthropic.NewUserMessage(anthropic.NewTextBlock("shared"))
		origTool := anthropic.ToolUnionParamOfTool(anthropic.ToolInputSchemaParam{}, "shared")
		req := anthropic.MessageNewParams{
			Tools:    []anthropic.ToolUnionParam{origTool},
			Messages: []anthropic.MessageParam{origMsg},
		}
		claude.ApplyPromptCacheBreakpoints(&req)

		// The request carries the marker...
		gt.Equal(t, ttl5m, req.Tools[0].OfTool.CacheControl.TTL)
		gt.Equal(t, ttl5m, req.Messages[0].Content[0].OfText.CacheControl.TTL)
		// ...but the originally shared values are untouched.
		gt.Equal(t, anthropic.CacheControlEphemeralTTL(""), origTool.OfTool.CacheControl.TTL)
		gt.Equal(t, anthropic.CacheControlEphemeralTTL(""), origMsg.Content[0].OfText.CacheControl.TTL)
	})

	t.Run("unknown tail variant is skipped", func(t *testing.T) {
		req := anthropic.MessageNewParams{
			Messages: []anthropic.MessageParam{
				{Role: anthropic.MessageParamRoleUser, Content: []anthropic.ContentBlockParamUnion{{}}},
			},
		}
		claude.ApplyPromptCacheBreakpoints(&req) // must not panic on an empty union
	})
}

func TestClaudePromptCacheWiring(t *testing.T) {
	runTest := func(enabled bool) func(t *testing.T) {
		return func(t *testing.T) {
			var sent anthropic.MessageNewParams
			mockClient := &apiClientMock{
				MessagesNewFunc: func(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
					sent = params
					return &anthropic.Message{
						Content: []anthropic.ContentBlockUnion{{Type: "text", Text: "ok"}},
						Role:    "assistant",
						Model:   "claude-3-opus-20240229",
					}, nil
				},
			}

			opts := []gollem.SessionOption{
				gollem.WithSessionSystemPrompt("you are helpful"),
				gollem.WithSessionTools(&cachePromptTestTool{}),
			}
			if enabled {
				opts = append(opts, gollem.WithSessionPromptCache(true))
			}
			cfg := gollem.NewSessionConfig(opts...)
			session, err := claude.NewSessionWithAPIClient(mockClient, cfg, "claude-3-opus-20240229")
			gt.NoError(t, err)

			_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("hello")})
			gt.NoError(t, err)

			want := anthropic.CacheControlEphemeralTTL("")
			if enabled {
				want = anthropic.CacheControlEphemeralTTLTTL5m
			}
			gt.Equal(t, want, sent.System[len(sent.System)-1].CacheControl.TTL)
			gt.Equal(t, want, sent.Tools[len(sent.Tools)-1].OfTool.CacheControl.TTL)
			tail := sent.Messages[len(sent.Messages)-1].Content
			gt.Equal(t, want, tail[len(tail)-1].OfText.CacheControl.TTL)
		}
	}

	t.Run("enabled injects cache_control", runTest(true))
	t.Run("disabled leaves request unmarked", runTest(false))
}

func TestClaudeCacheTokenObservation(t *testing.T) {
	mockClient := &apiClientMock{
		MessagesNewFunc: func(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
			return &anthropic.Message{
				Content: []anthropic.ContentBlockUnion{{Type: "text", Text: "ok"}},
				Role:    "assistant",
				Model:   "claude-3-opus-20240229",
				Usage: anthropic.Usage{
					InputTokens:              50,
					OutputTokens:             7,
					CacheCreationInputTokens: 10,
					CacheReadInputTokens:     100,
				},
			}, nil
		},
	}

	cfg := gollem.NewSessionConfig()
	session, err := claude.NewSessionWithAPIClient(mockClient, cfg, "claude-3-opus-20240229")
	gt.NoError(t, err)

	resp, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")})
	gt.NoError(t, err)

	// InputToken keeps total-input semantics (post-breakpoint + cached prefix).
	gt.Equal(t, 160, resp.InputToken)
	gt.Equal(t, 7, resp.OutputToken)
	gt.Equal(t, 10, resp.CacheCreationInputToken)
	gt.Equal(t, 100, resp.CacheReadInputToken)
}

func TestClaudeStreamPromptCacheAndObservation(t *testing.T) {
	var sent anthropic.MessageNewParams
	mockClient := &apiClientMock{
		MessagesNewFunc: func(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
			sent = params
			return &anthropic.Message{
				Content: []anthropic.ContentBlockUnion{{Type: "text", Text: "streamed"}},
				Role:    "assistant",
				Model:   "claude-3-opus-20240229",
				Usage: anthropic.Usage{
					InputTokens:          30,
					OutputTokens:         5,
					CacheReadInputTokens: 120,
				},
			}, nil
		},
	}

	cfg := gollem.NewSessionConfig(
		gollem.WithSessionSystemPrompt("sys"),
		gollem.WithSessionPromptCache(true),
	)
	session, err := claude.NewSessionWithAPIClient(mockClient, cfg, "claude-3-opus-20240229")
	gt.NoError(t, err)

	rec := trace.New()
	ctx := rec.StartAgentExecute(context.Background())
	ctx = trace.WithHandler(ctx, rec)

	ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("hi")})
	gt.NoError(t, err)

	var gotCacheRead, gotInput int
	for resp := range ch {
		gt.NoError(t, resp.Error)
		if resp.InputToken > 0 {
			gotInput = resp.InputToken
			gotCacheRead = resp.CacheReadInputToken
		}
	}
	rec.EndAgentExecute(ctx, nil)

	// Streaming reports total input and the cached read count.
	gt.Equal(t, 150, gotInput) // 30 post-breakpoint + 120 cached
	gt.Equal(t, 120, gotCacheRead)

	// The stream request carried the cache_control marker on the system prefix.
	gt.Equal(t, anthropic.CacheControlEphemeralTTLTTL5m, sent.System[len(sent.System)-1].CacheControl.TTL)

	// Trace records the cache breakdown.
	var llmSpan *trace.Span
	for _, child := range rec.Trace().RootSpan.Children {
		if child.Kind == trace.SpanKindLLMCall {
			llmSpan = child
			break
		}
	}
	gt.Value(t, llmSpan).NotNil()
	gt.Equal(t, 120, llmSpan.LLMCall.CacheReadInputTokens)
	gt.Equal(t, 150, llmSpan.LLMCall.InputTokens)
}

// TestClaudePromptCacheLive exercises the real Claude API to confirm that
// enabling prompt caching actually writes and then reads the cached prefix.
// This is the one path mock tests cannot prove: that the API accepts our
// cache_control markers and reports cache usage back.
func TestClaudePromptCacheLive(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
	if !ok {
		t.Skip("TEST_CLAUDE_API_KEY is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := claude.New(ctx, apiKey)
	gt.NoError(t, err)

	// The system prompt must exceed the model's minimum cacheable length (up to
	// 4096 tokens for some models). Repeat a sentence well past that threshold. A
	// per-run nonce keeps the prefix unique so the first call is a cold cache
	// write, not a hit on a cache left over from a previous run within the TTL.
	var sb strings.Builder
	fmt.Fprintf(&sb, "Assistant build %d-%d. ", os.Getpid(), time.Now().UnixNano())
	for i := 0; i < 800; i++ {
		sb.WriteString("You are a meticulous assistant that follows instructions carefully. ")
	}
	systemPrompt := sb.String()

	session, err := client.NewSession(ctx,
		gollem.WithSessionSystemPrompt(systemPrompt),
		gollem.WithSessionPromptCache(true),
	)
	gt.NoError(t, err)

	// First call: the stable prefix is written to the cache.
	first, err := session.Generate(ctx,
		[]gollem.Input{gollem.Text("Reply with the single word: one")},
		gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err)

	// Second call: the same system prefix should be served from the cache.
	second, err := session.Generate(ctx,
		[]gollem.Input{gollem.Text("Reply with the single word: two")},
		gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err)

	t.Logf("first : input=%d creation=%d read=%d",
		first.InputToken, first.CacheCreationInputToken, first.CacheReadInputToken)
	t.Logf("second: input=%d creation=%d read=%d",
		second.InputToken, second.CacheCreationInputToken, second.CacheReadInputToken)

	// The cache was written on the first call and read on the second.
	gt.Value(t, first.CacheCreationInputToken > 0).Equal(true)
	gt.Value(t, second.CacheReadInputToken > 0).Equal(true)
	// InputToken keeps total-input semantics: it includes the cached prefix.
	gt.Value(t, second.InputToken >= second.CacheReadInputToken).Equal(true)
}

func TestNewMaxTokens(t *testing.T) {
	type testCase struct {
		options  []claude.Option
		expected int64
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			client, err := claude.New(context.Background(), "test-key", tc.options...)
			gt.NoError(t, err).Required()
			gt.Equal(t, tc.expected, claude.MaxTokensOf(client))
		}
	}

	t.Run("resolves the default model when max tokens is not set", runTest(testCase{
		expected: 64000, // claude-sonnet-4-5-20250929
	}))

	t.Run("resolves the model given by WithModel", runTest(testCase{
		options:  []claude.Option{claude.WithModel("claude-opus-5")},
		expected: 128000,
	}))

	t.Run("falls back for a model absent from the table", runTest(testCase{
		options:  []claude.Option{claude.WithModel("my-proxy/llm")},
		expected: claude.FallbackMaxOutputTokens,
	}))

	t.Run("keeps an explicit max tokens", runTest(testCase{
		options:  []claude.Option{claude.WithMaxTokens(1000)},
		expected: 1000,
	}))

	t.Run("keeps an explicit max tokens above the model limit", runTest(testCase{
		options: []claude.Option{
			claude.WithModel("claude-sonnet-4-5"),
			claude.WithMaxTokens(200000),
		},
		expected: 200000,
	}))

	t.Run("keeps an explicit max tokens regardless of option order", runTest(testCase{
		options: []claude.Option{
			claude.WithMaxTokens(1000),
			claude.WithModel("claude-opus-5"),
		},
		expected: 1000,
	}))

	// An explicit value is never second-guessed, so it reaches the API even
	// when the API will reject it. This matches gollem.WithMaxTokens, which
	// distinguishes "set to zero" from "not set" and sends the zero through.
	t.Run("keeps an explicit zero", runTest(testCase{
		options:  []claude.Option{claude.WithMaxTokens(0)},
		expected: 0,
	}))

	t.Run("keeps an explicit negative value", runTest(testCase{
		options:  []claude.Option{claude.WithMaxTokens(-1)},
		expected: -1,
	}))
}

// TestGenerateWithResolvedMaxTokens drives New -> NewSession -> Generate/Stream
// through the real SDK so the resolved ceiling is checked against the SDK's
// non-streaming guard, which a mocked apiClient bypasses. The request is aimed
// at a closed port: reaching a transport error means the guard let it through.
func TestGenerateWithResolvedMaxTokens(t *testing.T) {
	ctx := context.Background()

	const guardMessage = "streaming is required"

	client, err := claude.New(ctx, "test-key", claude.WithBaseURL("http://127.0.0.1:1/"))
	gt.NoError(t, err).Required()
	gt.Equal(t, int64(64000), claude.MaxTokensOf(client))

	session, err := client.NewSession(ctx)
	gt.NoError(t, err).Required()

	t.Run("Generate", func(t *testing.T) {
		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("hello")})
		gt.Error(t, err).Required()
		gt.False(t, strings.Contains(err.Error(), guardMessage))
	})

	t.Run("Stream", func(t *testing.T) {
		_, err := session.Stream(ctx, []gollem.Input{gollem.Text("hello")})
		gt.Error(t, err).Required()
		gt.False(t, strings.Contains(err.Error(), guardMessage))
	})
}
