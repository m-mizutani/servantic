package gollem_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/m-mizutani/gt"
	openaiSDK "github.com/sashabaranov/go-openai"
	"google.golang.org/genai"
)

// marshalClaudeMessages renders messages the way the Anthropic SDK sends them, so tests
// compare what the API receives instead of the Go representation that produced it.
func marshalClaudeMessages(t *testing.T, messages []anthropic.MessageParam) []string {
	t.Helper()
	out := make([]string, 0, len(messages))
	for _, msg := range messages {
		data, err := msg.MarshalJSON()
		gt.NoError(t, err)
		out = append(out, string(data))
	}
	return out
}

func TestOpenAIToClaudeConversion(t *testing.T) {
	type testCase struct {
		name             string
		messages         []openaiSDK.ChatCompletionMessage
		expectedMessages []anthropic.MessageParam
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			// OpenAI → History
			historyFromOpenAI, err := openai.NewHistory(tc.messages)
			gt.NoError(t, err)

			// History → Claude
			claudeMsgs, err := claude.ToMessages(historyFromOpenAI)
			gt.NoError(t, err)

			// Compare the wire form rather than the Go values. A tool_use input is carried
			// as json.RawMessage so that the SDK encoder emits the arguments verbatim, so
			// two message lists that produce the same request can differ as Go structs.
			gt.Equal(t, marshalClaudeMessages(t, tc.expectedMessages), marshalClaudeMessages(t, claudeMsgs))
		}
	}

	t.Run("text messages with all fields", runTest(testCase{
		name: "text messages with all fields",
		messages: []openaiSDK.ChatCompletionMessage{
			{Role: "user", Content: "Hello", Name: "user1"},
			{Role: "assistant", Content: "Hi there!", Name: "assistant1"},
		},
		expectedMessages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Hello")),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("Hi there!")),
		},
	}))

	t.Run("tool calls with all fields", runTest(testCase{
		name: "tool calls with all fields",
		messages: []openaiSDK.ChatCompletionMessage{
			{Role: "user", Content: "What's the weather in Tokyo and London?"},
			{
				Role: "assistant",
				ToolCalls: []openaiSDK.ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: openaiSDK.FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location":"Tokyo","unit":"celsius"}`,
						},
					},
					{
						ID:   "call_456",
						Type: "function",
						Function: openaiSDK.FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location":"London","unit":"celsius"}`,
						},
					},
				},
			},
			{
				Role:       "tool",
				Content:    `{"temperature":25,"condition":"sunny","humidity":60}`,
				ToolCallID: "call_123",
				Name:       "get_weather",
			},
			{
				Role:       "tool",
				Content:    `{"temperature":15,"condition":"rainy","humidity":80}`,
				ToolCallID: "call_456",
				Name:       "get_weather",
			},
			{Role: "assistant", Content: "Tokyo is sunny at 25°C. London is rainy at 15°C."},
		},
		expectedMessages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("What's the weather in Tokyo and London?")),
			anthropic.NewAssistantMessage(
				anthropic.NewToolUseBlock("call_123", map[string]interface{}{
					"location": "Tokyo",
					"unit":     "celsius",
				}, "get_weather"),
				anthropic.NewToolUseBlock("call_456", map[string]interface{}{
					"location": "London",
					"unit":     "celsius",
				}, "get_weather"),
			),
			// Claude requires every tool_result answering one assistant turn to be in a
			// single user message, so the two OpenAI tool messages become one.
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock("call_123", `{"condition":"sunny","humidity":60,"temperature":25}`, false),
				anthropic.NewToolResultBlock("call_456", `{"condition":"rainy","humidity":80,"temperature":15}`, false),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("Tokyo is sunny at 25°C. London is rainy at 15°C.")),
		},
	}))

	t.Run("multi-content with images", runTest(testCase{
		name: "multi-content with images",
		messages: []openaiSDK.ChatCompletionMessage{
			{
				Role: "user",
				MultiContent: []openaiSDK.ChatMessagePart{
					{Type: "text", Text: "What's in this image?"},
					{
						Type: "image_url",
						ImageURL: &openaiSDK.ChatMessageImageURL{
							URL:    "data:image/png;base64,iVBORw0KGgo=",
							Detail: "high",
						},
					},
				},
			},
			{Role: "assistant", Content: "I see a cat in the image."},
		},
		expectedMessages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewTextBlock("What's in this image?"),
				anthropic.NewImageBlockBase64("image/png", "iVBORw0KGgo="),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("I see a cat in the image.")),
		},
	}))

	t.Run("system message", runTest(testCase{
		name: "system message",
		messages: []openaiSDK.ChatCompletionMessage{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi! How can I help you?"},
		},
		expectedMessages: []anthropic.MessageParam{
			// Claude merges system message into first user message with "\n\n" separator
			anthropic.NewUserMessage(
				anthropic.NewTextBlock("You are a helpful assistant.\n\n"),
				anthropic.NewTextBlock("Hello"),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("Hi! How can I help you?")),
		},
	}))

	t.Run("PDF content", runTest(testCase{
		name: "PDF content",
		messages: []openaiSDK.ChatCompletionMessage{
			{
				Role: "user",
				MultiContent: []openaiSDK.ChatMessagePart{
					{Type: "text", Text: "Analyze this PDF"},
					{
						Type: "image_url",
						ImageURL: &openaiSDK.ChatMessageImageURL{
							URL: "data:application/pdf;base64,JVBERi0xLjQgdGVzdA==",
						},
					},
				},
			},
			{Role: "assistant", Content: "This PDF contains test data."},
		},
		expectedMessages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewTextBlock("Analyze this PDF"),
				anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
					Data: "JVBERi0xLjQgdGVzdA==",
				}),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("This PDF contains test data.")),
		},
	}))

}

func TestClaudeToGeminiConversion(t *testing.T) {
	type testCase struct {
		name             string
		messages         []anthropic.MessageParam
		expectedMessages []*genai.Content
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			// Claude → History
			historyFromClaude, err := claude.NewHistory(tc.messages)
			gt.NoError(t, err)

			// History → Gemini
			geminiContents, err := gemini.ToContents(historyFromClaude)
			gt.NoError(t, err)

			// Verify Gemini contents
			gt.Equal(t, tc.expectedMessages, geminiContents)
		}
	}

	t.Run("text messages", runTest(testCase{
		name: "text messages",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Hello, how are you?")),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("I'm doing well, thank you!")),
		},
		expectedMessages: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "Hello, how are you?"}}},
			{Role: "model", Parts: []*genai.Part{{Text: "I'm doing well, thank you!"}}},
		},
	}))

	t.Run("tool use with multiple calls", runTest(testCase{
		name: "tool use with multiple calls",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Calculate 5+3 and 10*2")),
			anthropic.NewAssistantMessage(
				anthropic.NewToolUseBlock("toolu_123", map[string]interface{}{"expression": "5+3"}, "calculate"),
				anthropic.NewToolUseBlock("toolu_456", map[string]interface{}{"expression": "10*2"}, "calculate"),
			),
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock("toolu_123", `{"result":8}`, false),
				anthropic.NewToolResultBlock("toolu_456", `{"result":20}`, false),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("5+3 equals 8, and 10*2 equals 20.")),
		},
		expectedMessages: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "Calculate 5+3 and 10*2"}}},
			{
				Role: "model",
				Parts: []*genai.Part{
					// Claude's tool_use IDs must be propagated to Gemini for
					// Gemini 3.x strict id matching.
					{FunctionCall: &genai.FunctionCall{ID: "toolu_123", Name: "calculate", Args: map[string]any{"expression": "5+3"}}},
					{FunctionCall: &genai.FunctionCall{ID: "toolu_456", Name: "calculate", Args: map[string]any{"expression": "10*2"}}},
				},
			},
			{
				Role: "user",
				Parts: []*genai.Part{
					// Claude now parses JSON, so result is properly structured. The tool name is
					// recovered from the tool_use block with the same ID, since a Claude
					// tool_result carries none and Gemini requires one.
					{FunctionResponse: &genai.FunctionResponse{ID: "toolu_123", Name: "calculate", Response: map[string]any{"result": float64(8)}}},
					{FunctionResponse: &genai.FunctionResponse{ID: "toolu_456", Name: "calculate", Response: map[string]any{"result": float64(20)}}},
				},
			},
			{Role: "model", Parts: []*genai.Part{{Text: "5+3 equals 8, and 10*2 equals 20."}}},
		},
	}))

	t.Run("mixed content blocks", runTest(testCase{
		name: "mixed content blocks",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Tell me a joke and check the time")),
			anthropic.NewAssistantMessage(
				anthropic.NewTextBlock("Here's a joke: Why did the chicken cross the road?"),
				anthropic.NewToolUseBlock("toolu_789", map[string]interface{}{}, "get_current_time"),
			),
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock("toolu_789", `{"time":"14:30:00","timezone":"UTC"}`, false),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("It's currently 14:30 UTC.")),
		},
		expectedMessages: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "Tell me a joke and check the time"}}},
			{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "Here's a joke: Why did the chicken cross the road?"},
					{FunctionCall: &genai.FunctionCall{ID: "toolu_789", Name: "get_current_time", Args: map[string]any{}}},
				},
			},
			{
				Role: "user",
				Parts: []*genai.Part{
					// Claude parses JSON response
					{FunctionResponse: &genai.FunctionResponse{ID: "toolu_789", Name: "get_current_time", Response: map[string]any{"time": "14:30:00", "timezone": "UTC"}}},
				},
			},
			{Role: "model", Parts: []*genai.Part{{Text: "It's currently 14:30 UTC."}}},
		},
	}))

	t.Run("image content", runTest(testCase{
		name: "image content",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewTextBlock("Analyze this image"),
				anthropic.NewImageBlockBase64("image/jpeg", "/9j/4AAQSkZJRg=="),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("This appears to be a landscape photo.")),
		},
		expectedMessages: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Analyze this image"},
					{InlineData: &genai.Blob{MIMEType: "image/jpeg", Data: []byte{0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46}}},
				},
			},
			{Role: "model", Parts: []*genai.Part{{Text: "This appears to be a landscape photo."}}},
		},
	}))

	t.Run("error tool result", runTest(testCase{
		name: "error tool result",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Get the weather")),
			anthropic.NewAssistantMessage(
				anthropic.NewToolUseBlock("toolu_error", map[string]interface{}{"location": "InvalidCity"}, "get_weather"),
			),
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock("toolu_error", `{"error":"City not found"}`, true),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("I couldn't find that city.")),
		},
		expectedMessages: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "Get the weather"}}},
			{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "toolu_error", Name: "get_weather", Args: map[string]any{"location": "InvalidCity"}}},
				},
			},
			{
				Role: "user",
				Parts: []*genai.Part{
					// Error responses are also parsed as JSON
					{FunctionResponse: &genai.FunctionResponse{ID: "toolu_error", Name: "get_weather", Response: map[string]any{"error": "City not found"}}},
				},
			},
			{Role: "model", Parts: []*genai.Part{{Text: "I couldn't find that city."}}},
		},
	}))

	t.Run("PDF content", runTest(testCase{
		name: "PDF content",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewTextBlock("Analyze this PDF"),
				anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
					Data: base64.StdEncoding.EncodeToString([]byte("%PDF-1.4 test")),
				}),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("This PDF contains test data.")),
		},
		expectedMessages: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Analyze this PDF"},
					{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-1.4 test")}},
				},
			},
			{Role: "model", Parts: []*genai.Part{{Text: "This PDF contains test data."}}},
		},
	}))
}

func TestGeminiToOpenAIConversion(t *testing.T) {
	type testCase struct {
		name             string
		contents         []*genai.Content
		expectedMessages []openaiSDK.ChatCompletionMessage
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			// Gemini → History
			historyFromGemini, err := gemini.NewHistory(tc.contents)
			gt.NoError(t, err)

			// History → OpenAI
			openaiMsgs, err := openai.ToMessages(historyFromGemini)
			gt.NoError(t, err)

			// Verify OpenAI messages
			gt.Equal(t, tc.expectedMessages, openaiMsgs)
		}
	}

	t.Run("text messages", runTest(testCase{
		name: "text messages",
		contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "Hello from Gemini"}}},
			{Role: "model", Parts: []*genai.Part{{Text: "Hello! How can I assist you?"}}},
		},
		expectedMessages: []openaiSDK.ChatCompletionMessage{
			{Role: "user", Content: "Hello from Gemini"},
			{Role: "assistant", Content: "Hello! How can I assist you?"},
		},
	}))

	t.Run("function calls with complex args", runTest(testCase{
		name: "function calls with complex args",
		contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "Search for Python tutorials"}}},
			{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						Name: "search",
						Args: map[string]any{
							"query":  "Python tutorials",
							"limit":  float64(10),
							"filter": map[string]any{"language": "en", "level": "beginner"},
						},
					},
				}},
			},
			{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name: "search",
						Response: map[string]any{
							"results": []any{
								map[string]any{"title": "Python Basics", "url": "https://example.com/1"},
								map[string]any{"title": "Learn Python", "url": "https://example.com/2"},
							},
							"total": float64(2),
						},
					},
				}},
			},
			{Role: "model", Parts: []*genai.Part{{Text: "I found 2 Python tutorials for beginners."}}},
		},
		expectedMessages: []openaiSDK.ChatCompletionMessage{
			{Role: "user", Content: "Search for Python tutorials"},
			{
				Role: "assistant",
				ToolCalls: []openaiSDK.ToolCall{{
					ID:   "gemini-fallback-search-0",
					Type: "function",
					Function: openaiSDK.FunctionCall{
						Name:      "search",
						Arguments: `{"filter":{"language":"en","level":"beginner"},"limit":10,"query":"Python tutorials"}`,
					},
				}},
			},
			{
				Role:       "tool",
				Content:    `{"results":[{"title":"Python Basics","url":"https://example.com/1"},{"title":"Learn Python","url":"https://example.com/2"}],"total":2}`,
				ToolCallID: "gemini-fallback-search-0",
				Name:       "search",
			},
			{Role: "assistant", Content: "I found 2 Python tutorials for beginners."},
		},
	}))

	t.Run("multiple parts in single message", runTest(testCase{
		name: "multiple parts in single message",
		contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "First part"},
					{Text: "Second part"},
				},
			},
			{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "Response part 1"},
					{Text: "Response part 2"},
				},
			},
		},
		expectedMessages: []openaiSDK.ChatCompletionMessage{
			{
				Role: "user",
				MultiContent: []openaiSDK.ChatMessagePart{
					{Type: "text", Text: "First part"},
					{Type: "text", Text: "Second part"},
				},
			},
			{
				Role: "assistant",
				MultiContent: []openaiSDK.ChatMessagePart{
					{Type: "text", Text: "Response part 1"},
					{Type: "text", Text: "Response part 2"},
				},
			},
		},
	}))

	t.Run("PDF content", runTest(testCase{
		name: "PDF content",
		contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Analyze this PDF"},
					{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-1.4 test")}},
				},
			},
			{Role: "model", Parts: []*genai.Part{{Text: "This PDF contains test data."}}},
		},
		expectedMessages: []openaiSDK.ChatCompletionMessage{
			{
				Role: "user",
				MultiContent: []openaiSDK.ChatMessagePart{
					{Type: "text", Text: "Analyze this PDF"},
					{
						Type: "image_url",
						ImageURL: &openaiSDK.ChatMessageImageURL{
							URL: "data:application/pdf;base64,JVBERi0xLjQgdGVzdA==",
						},
					},
				},
			},
			{Role: "assistant", Content: "This PDF contains test data."},
		},
	}))
}

// Round-trip tests: A → B → A' should preserve A = A'
// Note: Some fields may be lost during conversion due to provider limitations:
// - Tool call IDs through Gemini (Gemini regenerates IDs)
func TestOpenAIRoundTrip(t *testing.T) {
	type testCase struct {
		name     string
		messages []openaiSDK.ChatCompletionMessage
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			// OpenAI → History → Claude → History → OpenAI
			historyFromOpenAI, err := openai.NewHistory(tc.messages)
			gt.NoError(t, err)

			claudeMsgs, err := claude.ToMessages(historyFromOpenAI)
			gt.NoError(t, err)

			historyFromClaude, err := claude.NewHistory(claudeMsgs)
			gt.NoError(t, err)

			restoredOpenAI, err := openai.ToMessages(historyFromClaude)
			gt.NoError(t, err)

			// A = A'
			gt.Equal(t, tc.messages, restoredOpenAI)
		}
	}

	t.Run("text messages", runTest(testCase{
		name: "text messages",
		messages: []openaiSDK.ChatCompletionMessage{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
		},
	}))

	t.Run("tool calls", runTest(testCase{
		name: "tool calls",
		messages: []openaiSDK.ChatCompletionMessage{
			{Role: "user", Content: "What's the weather?"},
			{
				Role: "assistant",
				ToolCalls: []openaiSDK.ToolCall{{
					ID:   "call_123",
					Type: "function",
					Function: openaiSDK.FunctionCall{
						Name:      "get_weather",
						Arguments: `{"location":"Tokyo"}`,
					},
				}},
			},
			{
				Role:       "tool",
				Content:    `{"temperature":25}`,
				ToolCallID: "call_123",
				// A Claude tool_result carries no tool name, so this used to come back empty.
				// It is now recovered from the tool_use block with the same ID.
				Name: "get_weather",
			},
			{Role: "assistant", Content: "It's 25°C in Tokyo."},
		},
	}))

	t.Run("PDF content", runTest(testCase{
		name: "PDF content",
		messages: []openaiSDK.ChatCompletionMessage{
			{
				Role: "user",
				MultiContent: []openaiSDK.ChatMessagePart{
					{Type: "text", Text: "Analyze this PDF"},
					{
						Type: "image_url",
						ImageURL: &openaiSDK.ChatMessageImageURL{
							URL: "data:application/pdf;base64,JVBERi0xLjQgdGVzdA==",
						},
					},
				},
			},
			{Role: "assistant", Content: "This PDF contains test data."},
		},
	}))
}

func TestClaudeRoundTrip(t *testing.T) {
	type testCase struct {
		name     string
		messages []anthropic.MessageParam
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			// Claude → History → Gemini → History → Claude
			historyFromClaude, err := claude.NewHistory(tc.messages)
			gt.NoError(t, err)

			geminiContents, err := gemini.ToContents(historyFromClaude)
			gt.NoError(t, err)

			historyFromGemini, err := gemini.NewHistory(geminiContents)
			gt.NoError(t, err)

			restoredClaude, err := claude.ToMessages(historyFromGemini)
			gt.NoError(t, err)

			// A = A'
			gt.Equal(t, tc.messages, restoredClaude)
		}
	}

	t.Run("text messages", runTest(testCase{
		name: "text messages",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Hello")),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("Hi!")),
		},
	}))

	t.Run("PDF content", runTest(testCase{
		name: "PDF content",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewTextBlock("Analyze this PDF"),
				anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
					Data: base64.StdEncoding.EncodeToString([]byte("%PDF-1.4 test")),
				}),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("This PDF contains test data.")),
		},
	}))

	// Note: Tool IDs cannot be perfectly round-tripped because Gemini
	// regenerates tool call IDs. Text-only messages can round-trip perfectly.
}

func TestGeminiRoundTrip(t *testing.T) {
	type testCase struct {
		name     string
		contents []*genai.Content
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			// Gemini → History → OpenAI → History → Gemini
			historyFromGemini, err := gemini.NewHistory(tc.contents)
			gt.NoError(t, err)

			openaiMsgs, err := openai.ToMessages(historyFromGemini)
			gt.NoError(t, err)

			historyFromOpenAI, err := openai.NewHistory(openaiMsgs)
			gt.NoError(t, err)

			restoredGemini, err := gemini.ToContents(historyFromOpenAI)
			gt.NoError(t, err)

			// A = A'
			gt.Equal(t, tc.contents, restoredGemini)
		}
	}

	t.Run("text messages", runTest(testCase{
		name: "text messages",
		contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "Hello"}}},
			{Role: "model", Parts: []*genai.Part{{Text: "Hi!"}}},
		},
	}))

	t.Run("function calls", runTest(testCase{
		name: "function calls",
		contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "Search Python"}}},
			{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						Name: "search",
						Args: map[string]any{"query": "Python"},
					},
				}},
			},
			{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name:     "search",
						Response: map[string]any{"results": []any{"Python tutorial"}},
					},
				}},
			},
			{Role: "model", Parts: []*genai.Part{{Text: "Found Python tutorial."}}},
		},
	}))

	t.Run("PDF content", runTest(testCase{
		name: "PDF content",
		contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Analyze this PDF"},
					{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-1.4 test")}},
				},
			},
			{Role: "model", Parts: []*genai.Part{{Text: "This PDF contains test data."}}},
		},
	}))
}

func TestHistoryUnmarshalVersionValidation(t *testing.T) {
	type testCase struct {
		version   int
		expectErr bool
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			data := fmt.Sprintf(`{"type":"OpenAI","version":%d,"messages":[]}`, tc.version)
			var h gollem.History
			err := json.Unmarshal([]byte(data), &h)

			if tc.expectErr {
				gt.Error(t, err)
				gt.True(t, errors.Is(err, gollem.ErrHistoryVersionMismatch))
			} else {
				gt.NoError(t, err)
				gt.Equal(t, tc.version, h.Version)
			}
		}
	}

	t.Run("current version", runTest(testCase{
		version:   gollem.HistoryVersion,
		expectErr: false,
	}))

	t.Run("old version 1", runTest(testCase{
		version:   1,
		expectErr: true,
	}))

	t.Run("old version 2", runTest(testCase{
		version:   2,
		expectErr: true,
	}))

	t.Run("future version", runTest(testCase{
		version:   99,
		expectErr: true,
	}))

	t.Run("zero version", runTest(testCase{
		version:   0,
		expectErr: true,
	}))
}

func TestHistoryCloneWithCurrentVersion(t *testing.T) {
	original := &gollem.History{
		LLType:  gollem.LLMTypeOpenAI,
		Version: gollem.HistoryVersion,
	}
	cloned := original.Clone()
	gt.Equal(t, original.LLType, cloned.LLType)
	gt.Equal(t, original.Version, cloned.Version)
}

// TestClonePreservesContentMeta verifies that History.Clone performs a true deep
// copy of every MessageContent field, including Meta (which carries Gemini's
// ThoughtSignature and Claude content-block metadata).
func TestClonePreservesContentMeta(t *testing.T) {
	meta := json.RawMessage(`{"thought_signature":"YWJjZA=="}`)
	original := &gollem.History{
		LLType:  gollem.LLMTypeGemini,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			{
				Role: gollem.RoleAssistant,
				Contents: []gollem.MessageContent{
					{
						Type: gollem.MessageContentTypeThinking,
						Data: json.RawMessage(`{"text":"reasoning"}`),
						Meta: meta,
					},
				},
			},
		},
	}

	cloned := original.Clone()

	// Data was already copied; Meta must be too.
	gt.Equal(t, original.Messages[0].Contents[0].Data, cloned.Messages[0].Contents[0].Data)
	gt.Equal(t, original.Messages[0].Contents[0].Meta, cloned.Messages[0].Contents[0].Meta)
}

// Clone used to deep-copy Metadata through a JSON round-trip, which dropped it entirely on
// an encoding failure and rewrote every number as a float64.
func TestCloneMetadataIsIndependentAndKeepsValues(t *testing.T) {
	original := &gollem.History{
		LLType:  gollem.LLMTypeClaude,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			{
				Role: gollem.RoleAssistant,
				Metadata: map[string]any{
					"account": int64(9007199254740993),
					"nested":  map[string]any{"tags": []any{"a", "b"}},
				},
			},
		},
	}

	cloned := original.Clone()

	gt.Equal(t, int64(9007199254740993), gt.Cast[int64](t, cloned.Messages[0].Metadata["account"]))

	// Mutating the clone must not reach the original.
	clonedNested := gt.Cast[map[string]any](t, cloned.Messages[0].Metadata["nested"])
	clonedTags := gt.Cast[[]any](t, clonedNested["tags"])
	clonedTags[0] = "changed"
	cloned.Messages[0].Metadata["account"] = int64(1)

	originalNested := gt.Cast[map[string]any](t, original.Messages[0].Metadata["nested"])
	originalTags := gt.Cast[[]any](t, originalNested["tags"])
	gt.Equal(t, "a", gt.Cast[string](t, originalTags[0]))
	gt.Equal(t, int64(9007199254740993), gt.Cast[int64](t, original.Messages[0].Metadata["account"]))
}
