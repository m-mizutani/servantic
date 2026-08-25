package gemini_test

import (
	"encoding/json"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/m-mizutani/gt"
	"google.golang.org/genai"
)

// normalizeGeminiMessages normalizes JSON content in function response parts
// to handle JSON key reordering during round-trip conversion
func normalizeGeminiMessages(contents []*genai.Content) []*genai.Content {
	result := make([]*genai.Content, len(contents))
	for i, content := range contents {
		normalizedParts := make([]*genai.Part, len(content.Parts))
		for j, part := range content.Parts {
			// Normalize FunctionResponse parts
			if part.FunctionResponse != nil && part.FunctionResponse.Response != nil {
				// Re-marshal to normalize JSON key order
				if data, err := json.Marshal(part.FunctionResponse.Response); err == nil {
					var normalized map[string]any
					if err := json.Unmarshal(data, &normalized); err == nil {
						normalizedParts[j] = &genai.Part{
							FunctionResponse: &genai.FunctionResponse{
								ID:       part.FunctionResponse.ID,
								Name:     part.FunctionResponse.Name,
								Response: normalized,
							},
						}
						continue
					}
				}
			}
			normalizedParts[j] = part
		}
		result[i] = &genai.Content{
			Role:  content.Role,
			Parts: normalizedParts,
		}
	}
	return result
}

func TestGeminiMessageRoundTrip(t *testing.T) {
	type testCase struct {
		name     string
		contents []*genai.Content
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			// Convert Gemini contents to gollem.History
			history, err := gemini.NewHistory(tc.contents)
			gt.NoError(t, err)

			// Convert back to Gemini contents
			restored, err := gemini.ToContents(history)
			gt.NoError(t, err)

			// Normalize JSON content before comparison
			normalizedOrig := normalizeGeminiMessages(tc.contents)
			normalizedRest := normalizeGeminiMessages(restored)

			// Compare normalized messages
			gt.Equal(t, normalizedOrig, normalizedRest)
		}
	}

	t.Run("text messages", runTest(testCase{
		name: "text messages",
		contents: []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: "Hello"}},
			},
			{
				Role:  "model",
				Parts: []*genai.Part{{Text: "Hi, how can I help you?"}},
			},
		},
	}))

	t.Run("function call and response", runTest(testCase{
		name: "function call and response",
		contents: []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: "What's the weather?"}},
			},
			{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						Name: "get_weather",
						Args: map[string]any{"location": "Tokyo"},
					},
				}},
			},
			{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name:     "get_weather",
						Response: map[string]any{"temperature": float64(25), "condition": "sunny"},
					},
				}},
			},
			{
				Role:  "model",
				Parts: []*genai.Part{{Text: "The weather in Tokyo is sunny with a temperature of 25°C."}},
			},
		},
	}))

	t.Run("multiple parts", runTest(testCase{
		name: "multiple parts",
		contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Tell me a joke"},
					{Text: "and the weather"},
				},
			},
			{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "Here's a joke: Why did the chicken cross the road?"},
					{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "London"},
						},
					},
				},
			},
		},
	}))

	t.Run("PDF inline data", runTest(testCase{
		name: "PDF inline data",
		contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Analyze this PDF"},
					{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("%PDF-1.4 test")}},
				},
			},
			{
				Role:  "model",
				Parts: []*genai.Part{{Text: "This PDF contains test data."}},
			},
		},
	}))

	t.Run("thought signature on function call", runTest(testCase{
		name: "thought signature on function call",
		contents: []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: "Write a file"}},
			},
			{
				Role: "model",
				Parts: []*genai.Part{
					{
						Text:             "Let me think about this...",
						Thought:          true,
						ThoughtSignature: []byte("thought-sig-001"),
					},
					{
						Text:             "I'll write the file for you.",
						ThoughtSignature: []byte("text-sig-002"),
					},
					{
						FunctionCall: &genai.FunctionCall{
							Name: "write_file",
							Args: map[string]any{"path": "test.txt", "content": "hello"},
						},
						ThoughtSignature: []byte("fc-sig-003"),
					},
				},
			},
		},
	}))

	t.Run("thought part only", runTest(testCase{
		name: "thought part only",
		contents: []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: "Hello"}},
			},
			{
				Role: "model",
				Parts: []*genai.Part{
					{
						Text:             "Internal reasoning...",
						Thought:          true,
						ThoughtSignature: []byte("thought-sig"),
					},
					{
						Text: "Hello! How can I help you?",
					},
				},
			},
		},
	}))
}

func TestThoughtSignatureRoundTrip(t *testing.T) {
	// Verify that ThoughtSignature is preserved through Gemini -> Message -> Gemini conversion
	contents := []*genai.Content{
		{
			Role:  "user",
			Parts: []*genai.Part{{Text: "Write a file"}},
		},
		{
			Role: "model",
			Parts: []*genai.Part{
				{
					Text:             "Thinking...",
					Thought:          true,
					ThoughtSignature: []byte("thought-sig-abc"),
				},
				{
					FunctionCall: &genai.FunctionCall{
						Name: "write_file",
						Args: map[string]any{"path": "test.txt"},
					},
					ThoughtSignature: []byte("fc-sig-def"),
				},
			},
		},
	}

	// Convert to gollem History
	history, err := gemini.NewHistory(contents)
	gt.NoError(t, err)

	// Verify the Messages have Meta set
	gt.A(t, history.Messages).Length(2)
	modelMsg := history.Messages[1]
	gt.A(t, modelMsg.Contents).Length(2)

	// Thought part should have Meta
	gt.Value(t, modelMsg.Contents[0].Meta).NotEqual(json.RawMessage(nil))

	// Function call part should have Meta
	gt.Value(t, modelMsg.Contents[1].Meta).NotEqual(json.RawMessage(nil))

	// Convert back to Gemini contents
	restored, err := gemini.ToContents(history)
	gt.NoError(t, err)

	gt.A(t, restored).Length(2)
	modelContent := restored[1]
	gt.A(t, modelContent.Parts).Length(2)

	// Verify thought part preserved
	thoughtPart := modelContent.Parts[0]
	gt.Value(t, thoughtPart.Thought).Equal(true)
	gt.Value(t, thoughtPart.ThoughtSignature).Equal([]byte("thought-sig-abc"))
	gt.Value(t, thoughtPart.Text).Equal("Thinking...")

	// Verify function call part preserved
	fcPart := modelContent.Parts[1]
	gt.Value(t, fcPart.ThoughtSignature).Equal([]byte("fc-sig-def"))
	gt.Value(t, fcPart.FunctionCall.Name).Equal("write_file")
}

func TestThoughtPartsExcludedFromResponse(t *testing.T) {
	// Verify that thought parts (Thought: true) are converted to ThinkingContentType.
	// This ensures proper separation of internal reasoning from visible response.

	// Convert a model message with thought parts to gollem format
	contents := []*genai.Content{
		{
			Role: "model",
			Parts: []*genai.Part{
				{
					Text:    "Internal reasoning",
					Thought: true,
				},
				{
					Text: "Visible response",
				},
			},
		},
	}

	history, err := gemini.NewHistory(contents)
	gt.NoError(t, err)

	// Both parts should be in the message (for history preservation)
	gt.A(t, history.Messages[0].Contents).Length(2)

	// First part should be ThinkingContentType with Meta
	thoughtContent := history.Messages[0].Contents[0]
	gt.Value(t, thoughtContent.Type).Equal(gollem.MessageContentTypeThinking)
	gt.Value(t, thoughtContent.Meta).NotEqual(json.RawMessage(nil))

	// Second part should be normal TextContentType without Meta
	normalContent := history.Messages[0].Contents[1]
	gt.Value(t, normalContent.Type).Equal(gollem.MessageContentTypeText)
	// Normal text without thought/signature should have nil Meta
	gt.Value(t, normalContent.Meta).Equal(json.RawMessage(nil))
}

func TestBackwardCompatibilityWithoutMeta(t *testing.T) {
	// Verify that messages without Meta field (from older versions) still work
	contents := []*genai.Content{
		{
			Role:  "user",
			Parts: []*genai.Part{{Text: "Hello"}},
		},
		{
			Role: "model",
			Parts: []*genai.Part{
				{Text: "Hi there!"},
				{
					FunctionCall: &genai.FunctionCall{
						Name: "greet",
						Args: map[string]any{"name": "world"},
					},
				},
			},
		},
	}

	// Convert to history (no ThoughtSignature anywhere)
	history, err := gemini.NewHistory(contents)
	gt.NoError(t, err)

	// All Meta should be nil
	for _, msg := range history.Messages {
		for _, content := range msg.Contents {
			gt.Value(t, content.Meta).Equal(json.RawMessage(nil))
		}
	}

	// Convert back should work without error
	restored, err := gemini.ToContents(history)
	gt.NoError(t, err)

	// Restored parts should have zero-value Thought/ThoughtSignature
	modelContent := restored[1]
	for _, part := range modelContent.Parts {
		gt.Value(t, part.Thought).Equal(false)
		gt.Value(t, part.ThoughtSignature).Equal([]byte(nil))
	}
}

// TestFunctionCallIDPreservedWithExplicitID verifies that Gemini 3.x style
// FunctionCall/FunctionResponse IDs are propagated through the gollem.History
// boundary in both directions. Without this, parallel tool calls under the
// strict-match contract would lose correlation.
func TestFunctionCallIDPreservedWithExplicitID(t *testing.T) {
	const callID = "call_abc123"

	contents := []*genai.Content{
		{
			Role:  "user",
			Parts: []*genai.Part{{Text: "What's the weather?"}},
		},
		{
			Role: "model",
			Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{
					ID:   callID,
					Name: "get_weather",
					Args: map[string]any{"location": "Tokyo"},
				},
			}},
		},
		{
			Role: "user",
			Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{
					ID:       callID,
					Name:     "get_weather",
					Response: map[string]any{"temperature": float64(25)},
				},
			}},
		},
	}

	history, err := gemini.NewHistory(contents)
	gt.NoError(t, err)

	// gollem.ToolCallContent.ID must mirror FunctionCall.ID.
	callContent, err := history.Messages[1].Contents[0].GetToolCallContent()
	gt.NoError(t, err)
	gt.Value(t, callContent.ID).Equal(callID)

	respContent, err := history.Messages[2].Contents[0].GetToolResponseContent()
	gt.NoError(t, err)
	gt.Value(t, respContent.ToolCallID).Equal(callID)

	// Round-trip back into Gemini parts and ensure IDs survive.
	restored, err := gemini.ToContents(history)
	gt.NoError(t, err)

	gt.Value(t, restored[1].Parts[0].FunctionCall.ID).Equal(callID)
	gt.Value(t, restored[2].Parts[0].FunctionResponse.ID).Equal(callID)
}

// TestFunctionCallIDBackwardCompatNoID verifies that legacy responses without
// FunctionCall.ID still work: gollem fabricates a stable fallback id internally
// and strips it on the way out so older Gemini models do not see a synthetic id.
func TestFunctionCallIDBackwardCompatNoID(t *testing.T) {
	contents := []*genai.Content{
		{
			Role: "model",
			Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{
					Name: "get_weather",
					Args: map[string]any{"location": "Tokyo"},
				},
			}},
		},
		{
			Role: "user",
			Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{
					Name:     "get_weather",
					Response: map[string]any{"temperature": float64(25)},
				},
			}},
		},
	}

	history, err := gemini.NewHistory(contents)
	gt.NoError(t, err)

	// Fallback id is populated for internal correlation.
	callContent, err := history.Messages[0].Contents[0].GetToolCallContent()
	gt.NoError(t, err)
	gt.Value(t, callContent.ID).NotEqual("")

	respContent, err := history.Messages[1].Contents[0].GetToolResponseContent()
	gt.NoError(t, err)
	gt.Value(t, respContent.ToolCallID).Equal(callContent.ID)

	// On the way out, the fallback id must be stripped so we do not feed Gemini
	// an id it never issued.
	restored, err := gemini.ToContents(history)
	gt.NoError(t, err)

	gt.Value(t, restored[0].Parts[0].FunctionCall.ID).Equal("")
	gt.Value(t, restored[1].Parts[0].FunctionResponse.ID).Equal("")
}

func TestToContentsLeavesHistoryUnchanged(t *testing.T) {
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
		LLType:  gollem.LLMTypeGemini,
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

	first, err := gemini.ToContents(history)
	gt.NoError(t, err)

	// The same History is converted again on every later request, so a conversion must not
	// consume the system message or duplicate the trailing message.
	gt.Equal(t, before, history.Messages)

	second, err := gemini.ToContents(history)
	gt.NoError(t, err)
	gt.Equal(t, first, second)
}

func TestToolResponsesInSeparateMessagesBecomeOneContent(t *testing.T) {
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
		LLType:  gollem.LLMTypeGemini,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{text}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{call1, call2}},
			// A runtime that runs the calls one at a time appends one message per result.
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp1}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp2}},
		},
	}

	contents, err := gemini.ToContents(history)
	gt.NoError(t, err)

	// Gemini requires the answering turn to carry as many functionResponse parts as the
	// call turn carried functionCall parts.
	gt.Equal(t, 3, len(contents))
	gt.Equal(t, "user", contents[2].Role)
	gt.Equal(t, 2, len(contents[2].Parts))
	gt.Value(t, contents[2].Parts[0].FunctionResponse.Name).Equal("alpha")
	gt.Value(t, contents[2].Parts[1].FunctionResponse.Name).Equal("beta")
}

// A tool argument wider than float64 must reach the Gemini request with the value the
// model sent. The genai SDK encodes the request with encoding/json, so the check is made
// on the same encoder that builds the request body.
func TestGeminiHistoryPreservesWideIntegers(t *testing.T) {
	const wide = "9007199254740993"

	call, err := gollem.NewToolCallContent("call_1", "lookup", map[string]any{"id": json.Number(wide)})
	gt.NoError(t, err)
	resp, err := gollem.NewToolResponseContent("call_1", "lookup", map[string]any{"account": json.Number(wide)}, false)
	gt.NoError(t, err)

	history := &gollem.History{
		LLType:  gollem.LLMTypeGemini,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{call}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp}},
		},
	}

	contents, err := gemini.ToContents(history)
	gt.NoError(t, err)

	encodedArgs, err := json.Marshal(contents[0].Parts[0].FunctionCall.Args)
	gt.NoError(t, err)
	gt.Equal(t, `{"id":`+wide+`}`, string(encodedArgs))

	encodedResp, err := json.Marshal(contents[1].Parts[0].FunctionResponse.Response)
	gt.NoError(t, err)
	gt.Equal(t, `{"account":`+wide+`}`, string(encodedResp))
}
