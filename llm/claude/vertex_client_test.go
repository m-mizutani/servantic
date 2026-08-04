package claude_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/m-mizutani/gt"
)

func TestNewWithVertex(t *testing.T) {
	ctx := context.Background()

	t.Run("missing projectID", func(t *testing.T) {
		client, err := claude.NewWithVertex(ctx, "us-central1", "")
		gt.Error(t, err)
		gt.Nil(t, client)
		gt.True(t, strings.Contains(err.Error(), "projectID is required"))
	})

	t.Run("missing region", func(t *testing.T) {
		client, err := claude.NewWithVertex(ctx, "", "test-project")
		gt.Error(t, err)
		gt.Nil(t, client)
		gt.True(t, strings.Contains(err.Error(), "region is required"))
	})

	t.Run("valid parameters with options", func(t *testing.T) {
		prj, ok := os.LookupEnv("TEST_CLAUDE_VERTEX_AI_PROJECT_ID")
		if !ok {
			t.Skip("TEST_CLAUDE_VERTEX_AI_PROJECT_ID is not set")
		}
		client, err := claude.NewWithVertex(ctx, "us-central1", prj,
			claude.WithVertexModel("claude-sonnet-4@20250514"),
			claude.WithVertexTemperature(0.5),
			claude.WithVertexTopP(0.8),
			claude.WithVertexMaxTokens(2048),
			claude.WithVertexSystemPrompt("You are a helpful assistant"),
		)

		// Note: This will likely fail in CI/testing without proper GCP credentials
		// but we're mainly testing the parameter validation and setup
		if err != nil {
			// Expected in test environment without GCP credentials
			gt.True(t, strings.Contains(err.Error(), "failed to") || strings.Contains(err.Error(), "auth"))
			return
		}

		// If it succeeds, validate the configuration
		gt.NotNil(t, client)
	})
}

// vertexTraceResponse is a fixed Anthropic message payload with a prompt-cache
// usage breakdown. The Vertex path calls anthropic.Client directly instead of the
// apiClient interface, so it is driven over the wire rather than with a mock.
const vertexTraceResponse = `{
  "id": "msg_test",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4@20250514",
  "content": [{"type": "text", "text": "ok"}],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 12,
    "output_tokens": 3,
    "cache_creation_input_tokens": 34,
    "cache_read_input_tokens": 56
  }
}`

func TestVertexGeneratePromptCacheTrace(t *testing.T) {
	type systemBlock struct {
		CacheControl map[string]any `json:"cache_control"`
	}
	type sentRequest struct {
		System []systemBlock `json:"system"`
	}

	sentCh := make(chan sentRequest, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var sent sentRequest
		if err := json.NewDecoder(r.Body).Decode(&sent); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		sentCh <- sent

		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(vertexTraceResponse)); err != nil {
			// The client has already gone away, so nothing can be recovered here.
			// A truncated response surfaces as an error from Generate, which the
			// assertions below catch.
			return
		}
	}))
	defer srv.Close()

	client := anthropic.NewClient(
		option.WithBaseURL(srv.URL),
		option.WithAPIKey("test"),
	)
	cfg := gollem.NewSessionConfig(
		gollem.WithSessionSystemPrompt("sys"),
		gollem.WithSessionPromptCache(true),
	)
	session, err := claude.NewVertexSessionWithClient(&client, cfg, "claude-sonnet-4@20250514")
	gt.NoError(t, err)

	h := newCaptureHandler()
	ctx := h.start(context.Background())

	resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
	gt.NoError(t, err)

	gt.Equal(t, 102, resp.InputToken) // 12 post-breakpoint + 34 written + 56 cached
	gt.Equal(t, 3, resp.OutputToken)
	gt.Equal(t, 34, resp.CacheCreationInputToken)
	gt.Equal(t, 56, resp.CacheReadInputToken)

	// The registered trace handler receives the same breakdown.
	gt.Equal(t, 1, h.endCalls)
	gt.Nil(t, h.llmErr)
	data := h.llmCallData(t)
	gt.Equal(t, 102, data.InputTokens)
	gt.Equal(t, 3, data.OutputTokens)
	gt.Equal(t, 34, data.CacheCreationInputTokens)
	gt.Equal(t, 56, data.CacheReadInputTokens)

	// The request carried the cache_control marker on the system prefix.
	sent := <-sentCh
	if len(sent.System) == 0 {
		t.Fatal("request carried no system prompt")
	}
	gt.NotNil(t, sent.System[len(sent.System)-1].CacheControl)
}

func TestVertexClient(t *testing.T) {
	projectID := os.Getenv("TEST_CLAUDE_VERTEX_AI_PROJECT_ID")
	if projectID == "" {
		t.Skip("TEST_CLAUDE_VERTEX_AI_PROJECT_ID not set, skipping test")
	}

	location := os.Getenv("TEST_CLAUDE_VERTEX_AI_LOCATION")
	if location == "" {
		location = "us-east5" // Default to us-east5 where Claude Sonnet 4 is working
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	// Create Vertex AI client using Anthropic's official SDK
	client, err := claude.NewWithVertex(ctx, location, projectID,
		claude.WithVertexModel("claude-sonnet-4@20250514"),
		claude.WithVertexMaxTokens(512),
		claude.WithVertexTemperature(0.5),
	)
	gt.NoError(t, err)

	// Create session
	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	// Test basic text generation
	response, err := session.Generate(ctx, []gollem.Input{gollem.Text("Hello! Please respond with 'Vertex AI working!' to confirm this integration works.")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err)
	gt.NotNil(t, response)
	gt.True(t, len(response.Texts) > 0)

	gt.True(t, strings.Contains(response.Texts[0], "Vertex AI working!"))
}

func TestVertexClientWithTools(t *testing.T) {
	projectID := os.Getenv("TEST_CLAUDE_VERTEX_AI_PROJECT_ID")
	if projectID == "" {
		t.Skip("TEST_CLAUDE_VERTEX_AI_PROJECT_ID not set, skipping test")
	}

	location := os.Getenv("TEST_CLAUDE_VERTEX_AI_LOCATION")
	if location == "" {
		location = "us-east5" // Default to us-east5 where Claude Sonnet 4 is working
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	// Create Vertex AI client using Anthropic's official SDK
	client, err := claude.NewWithVertex(ctx, location, projectID,
		claude.WithVertexModel("claude-sonnet-4@20250514"),
		claude.WithVertexMaxTokens(512),
		claude.WithVertexTemperature(0.5),
	)
	gt.NoError(t, err)

	// Create simple test tool
	testTool := &calculatorTool{}

	// Create session with tool
	session, err := client.NewSession(ctx, gollem.WithSessionTools(testTool))
	gt.NoError(t, err)

	// Test tool calling
	response, err := session.Generate(ctx, []gollem.Input{gollem.Text("Please calculate 15 + 27 using the calculator tool.")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err)
	gt.NotNil(t, response)

	// Should either return text or function calls
	gt.True(t, len(response.Texts) > 0 || len(response.FunctionCalls) > 0)

	if len(response.FunctionCalls) > 0 {

		// Execute the function call
		result, err := testTool.Run(ctx, response.FunctionCalls[0].Arguments)
		gt.NoError(t, err)

		// Send result back
		funcResp := gollem.FunctionResponse{
			ID:   response.FunctionCalls[0].ID,
			Name: response.FunctionCalls[0].Name,
			Data: result,
		}

		finalResponse, err := session.Generate(ctx, []gollem.Input{funcResp}, gollem.WithMaxTokens(maxTestTokens))
		gt.NoError(t, err)
		gt.NotNil(t, finalResponse)
		gt.True(t, len(finalResponse.Texts) > 0)

		gt.True(t, strings.Contains(finalResponse.Texts[0], "42"))
	}
}

// calculatorTool is a simple tool for testing
type calculatorTool struct{}

func (c *calculatorTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "calculator",
		Description: "Perform basic arithmetic operations",
		Parameters: map[string]*gollem.Parameter{
			"operation": {
				Type:        gollem.TypeString,
				Description: "The operation to perform (add, subtract, multiply, divide)",
				Required:    true,
			},
			"a": {
				Type:        gollem.TypeNumber,
				Description: "First number",
				Required:    true,
			},
			"b": {
				Type:        gollem.TypeNumber,
				Description: "Second number",
				Required:    true,
			},
		},
	}
}

func (c *calculatorTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	operation, ok := args["operation"].(string)
	if !ok {
		return nil, gollem.ErrInvalidParameter
	}

	a, ok := args["a"].(float64)
	if !ok {
		return nil, gollem.ErrInvalidParameter
	}

	b, ok := args["b"].(float64)
	if !ok {
		return nil, gollem.ErrInvalidParameter
	}

	var result float64
	switch operation {
	case "add":
		result = a + b
	case "subtract":
		result = a - b
	case "multiply":
		result = a * b
	case "divide":
		if b == 0 {
			return map[string]any{"error": "division by zero"}, nil
		}
		result = a / b
	default:
		return nil, gollem.ErrInvalidParameter
	}

	return map[string]any{"result": result}, nil
}
