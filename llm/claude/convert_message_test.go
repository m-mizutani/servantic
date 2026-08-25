package claude_test

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/m-mizutani/gt"
)

// marshalMessages renders messages the way the Anthropic SDK sends them, so tests compare
// what the API receives instead of the Go representation that produced it.
func marshalMessages(t *testing.T, messages []anthropic.MessageParam) []string {
	t.Helper()
	out := make([]string, 0, len(messages))
	for _, msg := range messages {
		data, err := msg.MarshalJSON()
		gt.NoError(t, err)
		out = append(out, string(data))
	}
	return out
}

func TestClaudeMessageRoundTrip(t *testing.T) {
	type testCase struct {
		name     string
		messages []anthropic.MessageParam
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			// Convert Claude messages to gollem.History
			history, err := claude.NewHistory(tc.messages)
			gt.NoError(t, err)

			// Convert back to Claude messages
			restored, err := claude.ToMessages(history)
			gt.NoError(t, err)

			// Compare the wire form rather than the Go values. A tool_use input is carried
			// as json.RawMessage so that the SDK encoder emits the arguments verbatim, so
			// two message lists that produce the same request can differ as Go structs.
			gt.Equal(t, marshalMessages(t, tc.messages), marshalMessages(t, restored))
		}
	}

	t.Run("text messages", runTest(testCase{
		name: "text messages",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Hello")),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("Hi, how can I help you?")),
		},
	}))

	t.Run("tool use and results", runTest(testCase{
		name: "tool use and results",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("What's the weather?")),
			anthropic.NewAssistantMessage(
				anthropic.NewToolUseBlock(
					"toolu_abc123",
					map[string]interface{}{"location": "Tokyo"},
					"get_weather",
				),
			),
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(
					"toolu_abc123",
					// JSON keys are alphabetically sorted after round-trip
					`{"condition":"sunny","temperature":25}`,
					false,
				),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("The weather in Tokyo is sunny with a temperature of 25°C.")),
		},
	}))

	t.Run("multiple content blocks", runTest(testCase{
		name: "multiple content blocks",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Tell me a joke and the weather")),
			anthropic.NewAssistantMessage(
				anthropic.NewTextBlock("Here's a joke: Why did the chicken cross the road? Let me check the weather..."),
				anthropic.NewToolUseBlock(
					"toolu_xyz789",
					map[string]interface{}{"location": "London"},
					"get_weather",
				),
			),
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(
					"toolu_xyz789",
					// JSON keys are alphabetically sorted after round-trip
					`{"condition":"rainy","temperature":15}`,
					false,
				),
			),
		},
	}))

	t.Run("PDF document block", runTest(testCase{
		name: "PDF document block",
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

	t.Run("thinking block", func(t *testing.T) {
		// Test thinking content conversion (Claude → gollem)
		block := anthropic.NewThinkingBlock("sig-123", "Let me think...")

		history, err := claude.NewHistory([]anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Help me")),
			anthropic.NewAssistantMessage(block),
		})
		gt.NoError(t, err)

		// Find assistant message with thinking content
		var assistantMsg *gollem.Message
		for i := range history.Messages {
			if history.Messages[i].Role == gollem.RoleAssistant {
				assistantMsg = &history.Messages[i]
				break
			}
		}

		gt.NotNil(t, assistantMsg)
		gt.Equal(t, 1, len(assistantMsg.Contents))

		content := assistantMsg.Contents[0]
		gt.Equal(t, gollem.MessageContentTypeThinking, content.Type)

		reasoning, err := content.GetThinkingContent()
		gt.NoError(t, err)
		gt.Equal(t, "Let me think...", reasoning.Text)
	})

	t.Run("redacted thinking block", func(t *testing.T) {
		// Test redacted thinking content conversion
		block := anthropic.NewRedactedThinkingBlock("Redacted")

		history, err := claude.NewHistory([]anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Help me")),
			anthropic.NewAssistantMessage(block),
		})
		gt.NoError(t, err)

		// Find assistant message with redacted thinking content
		var assistantMsg *gollem.Message
		for i := range history.Messages {
			if history.Messages[i].Role == gollem.RoleAssistant {
				assistantMsg = &history.Messages[i]
				break
			}
		}

		gt.NotNil(t, assistantMsg)
		gt.Equal(t, 1, len(assistantMsg.Contents))

		content := assistantMsg.Contents[0]
		gt.Equal(t, gollem.MessageContentTypeThinking, content.Type)

		reasoning, err := content.GetThinkingContent()
		gt.NoError(t, err)
		gt.Equal(t, "", reasoning.Text)
		// Signature is stored in meta
		gt.NotNil(t, content.Meta)
	})
}

func TestToMessagesLeavesHistoryUnchanged(t *testing.T) {
	sys, err := gollem.NewTextContent("be brief")
	gt.NoError(t, err)
	user, err := gollem.NewTextContent("hello")
	gt.NoError(t, err)
	assistant, err := gollem.NewTextContent("hi")
	gt.NoError(t, err)
	resp1, err := gollem.NewToolResponseContent("c1", "alpha", map[string]any{"ok": true}, false)
	gt.NoError(t, err)
	resp2, err := gollem.NewToolResponseContent("c2", "beta", map[string]any{"ok": true}, false)
	gt.NoError(t, err)

	history := &gollem.History{
		LLType:  gollem.LLMTypeClaude,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			{Role: gollem.RoleSystem, Contents: []gollem.MessageContent{sys}},
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{user}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{assistant}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp1}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp2}},
		},
	}
	before := append([]gollem.Message(nil), history.Messages...)

	first, err := claude.ToMessages(history)
	gt.NoError(t, err)

	// The same History is converted again on every later request, so a conversion must not
	// consume the system message or duplicate the trailing message.
	gt.Equal(t, before, history.Messages)

	second, err := claude.ToMessages(history)
	gt.NoError(t, err)
	gt.Equal(t, first, second)
}

func TestToolResponsesInSeparateMessagesBecomeOneMessage(t *testing.T) {
	text, err := gollem.NewTextContent("go")
	gt.NoError(t, err)
	call1, err := gollem.NewToolCallContent("c1", "alpha", map[string]any{"x": 1})
	gt.NoError(t, err)
	call2, err := gollem.NewToolCallContent("c2", "beta", map[string]any{"y": 2})
	gt.NoError(t, err)
	resp1, err := gollem.NewToolResponseContent("c1", "alpha", map[string]any{"ok": true}, false)
	gt.NoError(t, err)
	resp2, err := gollem.NewToolResponseContent("c2", "beta", map[string]any{"ok": true}, false)
	gt.NoError(t, err)

	history := &gollem.History{
		LLType:  gollem.LLMTypeClaude,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{text}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{call1, call2}},
			// A runtime that runs the calls one at a time appends one message per result.
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp1}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp2}},
		},
	}

	messages, err := claude.ToMessages(history)
	gt.NoError(t, err)

	// Claude requires one tool_result for each tool_use block, all in the next user message.
	gt.Equal(t, 3, len(messages))
	gt.Equal(t, anthropic.MessageParamRoleUser, messages[2].Role)
	gt.Equal(t, 2, len(messages[2].Content))
	gt.NotNil(t, messages[2].Content[0].OfToolResult)
	gt.Value(t, messages[2].Content[0].OfToolResult.ToolUseID).Equal("c1")
	gt.NotNil(t, messages[2].Content[1].OfToolResult)
	gt.Value(t, messages[2].Content[1].OfToolResult.ToolUseID).Equal("c2")
}

// A Claude tool_result block carries no tool name. Gemini requires one on every
// functionResponse part, so the name is recovered from the tool_use block with the same ID.
func TestNewHistoryRecoversToolNameForToolResults(t *testing.T) {
	messages := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock("what is the weather")),
		anthropic.NewAssistantMessage(
			anthropic.NewToolUseBlock("call_1", map[string]any{"city": "Tokyo"}, "get_weather"),
		),
		anthropic.NewUserMessage(
			anthropic.NewToolResultBlock("call_1", `{"temperature":25}`, false),
		),
	}

	history, err := claude.NewHistory(messages)
	gt.NoError(t, err)

	resp, err := history.Messages[2].Contents[0].GetToolResponseContent()
	gt.NoError(t, err)
	gt.Equal(t, "call_1", resp.ToolCallID)
	gt.Equal(t, "get_weather", resp.Name)
}

// A tool_result for a tool_use that is not in the message list has no name to recover.
// The response must still convert rather than fail.
func TestNewHistoryToolResultWithoutMatchingToolUse(t *testing.T) {
	messages := []anthropic.MessageParam{
		anthropic.NewUserMessage(
			anthropic.NewToolResultBlock("orphan", `{"ok":true}`, false),
		),
	}

	history, err := claude.NewHistory(messages)
	gt.NoError(t, err)

	resp, err := history.Messages[0].Contents[0].GetToolResponseContent()
	gt.NoError(t, err)
	gt.Equal(t, "orphan", resp.ToolCallID)
	gt.Equal(t, "", resp.Name)
}

// Tool arguments and results wider than float64 must reach the Anthropic request with the
// value the model sent. The Anthropic SDK marshals tool_use input through its own encoder,
// so this exercises the whole path rather than encoding/json alone.
func TestClaudeHistoryPreservesWideIntegers(t *testing.T) {
	const wide = "9007199254740993"

	messages := []anthropic.MessageParam{
		anthropic.NewAssistantMessage(
			anthropic.NewToolUseBlock("call_1", map[string]any{"id": json.Number(wide)}, "lookup"),
		),
		anthropic.NewUserMessage(
			anthropic.NewToolResultBlock("call_1", `{"account":9007199254740993}`, false),
		),
	}

	history, err := claude.NewHistory(messages)
	gt.NoError(t, err)

	call, err := history.Messages[0].Contents[0].GetToolCallContent()
	gt.NoError(t, err)
	encodedArgs, err := json.Marshal(call.Arguments)
	gt.NoError(t, err)
	gt.Equal(t, `{"id":`+wide+`}`, string(encodedArgs))

	resp, err := history.Messages[1].Contents[0].GetToolResponseContent()
	gt.NoError(t, err)
	encodedResp, err := json.Marshal(resp.Response)
	gt.NoError(t, err)
	gt.Equal(t, `{"account":`+wide+`}`, string(encodedResp))

	// Back into Claude's own request types, then out through the SDK's encoder.
	// anthropic.MessageParam has its own MarshalJSON, so this is the wire form the
	// Anthropic API actually receives, not encoding/json's view of it.
	restored, err := claude.ToMessages(history)
	gt.NoError(t, err)
	wire, err := restored[0].MarshalJSON()
	gt.NoError(t, err)
	gt.S(t, string(wire)).Contains(`"id":` + wide)
}

// Every site that builds a tool_use block must encode the arguments the same way. Handing
// the SDK a map instead makes its encoder write a json.Number as a quoted string, so a
// wide integer argument would leave the streaming path as a string while the history path
// sent a number.
func TestToolUseInputSendsNumbersAsNumbers(t *testing.T) {
	const wide = "9007199254740993"

	input, err := claude.ToolUseInput(map[string]any{"id": json.Number(wide)})
	gt.NoError(t, err)

	msg := anthropic.NewAssistantMessage(anthropic.NewToolUseBlock("call_1", input, "lookup"))
	wire, err := msg.MarshalJSON()
	gt.NoError(t, err)
	gt.S(t, string(wire)).Contains(`"id":` + wide)
	gt.S(t, string(wire)).NotContains(`"id":"` + wide)
}

// The trace handler reads tool_use arguments back out of the request blocks. Those blocks
// now carry json.RawMessage, so a type assertion for a map alone would trace every
// history-replayed tool call with no arguments at all.
func TestTraceMessagesReadRawToolUseInput(t *testing.T) {
	input, err := claude.ToolUseInput(map[string]any{"city": "Tokyo"})
	gt.NoError(t, err)

	messages := []anthropic.MessageParam{
		anthropic.NewAssistantMessage(anthropic.NewToolUseBlock("call_1", input, "get_weather")),
	}

	traced := claude.ClaudeMessagesToTraceMessages(messages)
	gt.A(t, traced).Length(1)
	gt.A(t, traced[0].Contents).Length(1)
	gt.Equal(t, "get_weather", traced[0].Contents[0].Name)
	gt.Equal(t, "Tokyo", gt.Cast[string](t, traced[0].Contents[0].Arguments["city"]))
}
