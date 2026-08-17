package gemini

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/internal/convert"
	"github.com/m-mizutani/goerr/v2"
	"google.golang.org/genai"
)

// geminiFallbackIDPrefix marks tool-call ids that gollem fabricated because
// the Gemini API did not supply one. Carrying such ids back to Gemini 3.x —
// which enforces strict call/response id matching — would feed it ids it
// never issued, so any id with this prefix is stripped on the way out.
const geminiFallbackIDPrefix = "gemini-fallback-"

// geminiFallbackToolCallID builds a fallback id used when the Gemini API
// returned a FunctionCall without an id. The same format must be produced by
// every code path that synthesizes an id, otherwise the strip logic below
// will leak fabricated ids back to the API.
func geminiFallbackToolCallID(name string, index int) string {
	return geminiFallbackIDPrefix + name + "-" + strconv.Itoa(index)
}

// isGeminiFallbackToolCallID reports whether id was synthesized by gollem and
// should be stripped before being sent back to Gemini.
func isGeminiFallbackToolCallID(id string) bool {
	return strings.HasPrefix(id, geminiFallbackIDPrefix)
}

// partMeta is the metadata stored in MessageContent.Meta for Gemini parts.
// It preserves Gemini-specific fields (e.g., thinking model signatures) across
// serialization/deserialization without polluting the common message types.
type partMeta struct {
	Thought          bool   `json:"thought,omitempty"`
	ThoughtSignature []byte `json:"thought_signature,omitempty"`
}

// marshalPartMeta marshals partMeta to JSON for MessageContent.Meta.
// Returns nil if no metadata needs to be stored.
func marshalPartMeta(m partMeta) (json.RawMessage, error) {
	if !m.Thought && len(m.ThoughtSignature) == 0 {
		return nil, nil
	}
	data, err := json.Marshal(m)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to marshal part meta")
	}
	return data, nil
}

// unmarshalPartMeta unmarshals partMeta from MessageContent.Meta.
// Returns zero-value partMeta if meta is nil or empty.
func unmarshalPartMeta(meta json.RawMessage) (partMeta, error) {
	if len(meta) == 0 {
		return partMeta{}, nil
	}
	var m partMeta
	if err := json.Unmarshal(meta, &m); err != nil {
		return partMeta{}, goerr.Wrap(err, "failed to unmarshal part meta")
	}
	return m, nil
}

// hasNonTextContent reports whether a part carries content beyond plain text.
// Every such field has to stay bound to the exact part that carries it, so this
// single list is what both isEmptyPart and isTextOnlyPart are defined against.
func hasNonTextContent(part *genai.Part) bool {
	return part.FunctionCall != nil ||
		part.FunctionResponse != nil ||
		part.InlineData != nil ||
		part.FileData != nil ||
		part.ExecutableCode != nil ||
		part.CodeExecutionResult != nil ||
		len(part.ThoughtSignature) > 0
}

// isEmptyPart returns true if a Gemini part has no meaningful content.
// Some models (e.g., thinking models) may return empty parts that should be skipped.
func isEmptyPart(part *genai.Part) bool {
	return part.Text == "" && !part.Thought && !hasNonTextContent(part)
}

// isConvertiblePart reports whether convertGeminiPart can represent the part in
// the portable history format.
func isConvertiblePart(part *genai.Part) bool {
	return part.Thought ||
		part.Text != "" ||
		part.InlineData != nil ||
		part.FileData != nil ||
		part.FunctionCall != nil ||
		part.FunctionResponse != nil ||
		len(part.ThoughtSignature) > 0
}

// newHistoryContent normalizes a model response before it is stored in the
// session history. Returns nil when nothing is left worth storing.
func newHistoryContent(content *genai.Content) *genai.Content {
	if content == nil {
		return nil
	}

	parts := make([]*genai.Part, 0, len(content.Parts))
	for _, part := range content.Parts {
		// A thought summary only has to travel back to Gemini as its signature.
		// Keeping the reasoning text would re-send it on every later turn, and
		// it would round-trip as a thinking block whose signature no other
		// provider can read once the history is handed over.
		if part.Thought {
			if len(part.ThoughtSignature) > 0 {
				parts = append(parts, &genai.Part{ThoughtSignature: part.ThoughtSignature})
			}
			continue
		}
		// A part convertGeminiPart cannot represent (executable code, code
		// execution results) would make every later History() call fail on the
		// whole session, so it is left out instead of stored unconvertible.
		if !isConvertiblePart(part) {
			continue
		}
		parts = append(parts, part)
	}

	if len(parts) == 0 {
		return nil
	}
	return &genai.Content{
		Role:  content.Role,
		Parts: parts,
	}
}

// isTextOnlyPart reports whether a part carries text and nothing else, making
// it safe to concatenate with an adjacent text part.
func isTextOnlyPart(part *genai.Part) bool {
	return part.Text != "" && !hasNonTextContent(part)
}

// mergeStreamedParts joins consecutive text-only parts of a streamed response
// into a single part. Streaming delivers text in many small deltas, and keeping
// each delta as its own history part would multiply the serialized history for
// no gain. Parts carrying a thought signature or a function call are passed
// through untouched so Gemini 3.x can still match them on the next turn.
func mergeStreamedParts(parts []*genai.Part) []*genai.Part {
	merged := make([]*genai.Part, 0, len(parts))

	var buf strings.Builder
	var bufThought bool
	flush := func() {
		if buf.Len() == 0 {
			return
		}
		merged = append(merged, &genai.Part{Text: buf.String(), Thought: bufThought})
		buf.Reset()
	}

	for _, part := range parts {
		if isTextOnlyPart(part) {
			// Thought text and answer text are distinct kinds of content and
			// must not be concatenated into one part.
			if buf.Len() > 0 && bufThought != part.Thought {
				flush()
			}
			bufThought = part.Thought
			buf.WriteString(part.Text)
			continue
		}
		flush()
		merged = append(merged, part)
	}
	flush()

	return merged
}

// convertGeminiToMessages converts Gemini contents to common Message format
func convertGeminiToMessages(contents []*genai.Content) ([]gollem.Message, error) {
	if len(contents) == 0 {
		return []gollem.Message{}, nil
	}

	result := make([]gollem.Message, 0, len(contents))

	for _, content := range contents {
		msg, err := convertGeminiContent(content)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert Gemini content")
		}
		result = append(result, msg)
	}

	return result, nil
}

// convertGeminiContent converts a single Gemini content to Message
func convertGeminiContent(content *genai.Content) (gollem.Message, error) {
	contents := make([]gollem.MessageContent, 0, len(content.Parts))

	// Index used to disambiguate fallback ids for FunctionCall/Response parts
	// within a single content. Without it, two same-name calls without ids
	// would collide on the same synthesized id.
	toolPartIndex := 0
	for _, part := range content.Parts {
		if isEmptyPart(part) {
			continue
		}
		msgContent, err := convertGeminiPart(part, toolPartIndex)
		if err != nil {
			return gollem.Message{}, goerr.Wrap(err, "failed to convert Gemini part")
		}
		if part.FunctionCall != nil || part.FunctionResponse != nil {
			toolPartIndex++
		}
		contents = append(contents, msgContent)
	}

	// Convert role - model is always converted to assistant
	role := convert.ConvertRoleToCommon(content.Role)

	return gollem.Message{
		Role:     role,
		Contents: contents,
	}, nil
}

// convertGeminiPart converts a Gemini part to MessageContent. fallbackIndex
// disambiguates synthesized tool-call ids for parts whose FunctionCall.ID is
// empty.
func convertGeminiPart(part *genai.Part, fallbackIndex int) (gollem.MessageContent, error) {
	// Build metadata from thinking-related fields
	meta, err := marshalPartMeta(partMeta{
		Thought:          part.Thought,
		ThoughtSignature: part.ThoughtSignature,
	})
	if err != nil {
		return gollem.MessageContent{}, err
	}

	// Thought part (internal reasoning, may have empty text)
	if part.Thought {
		mc, err := gollem.NewThinkingContent(part.Text)
		if err != nil {
			return gollem.MessageContent{}, err
		}
		mc.Meta = meta
		return mc, nil
	}

	// Text part
	if part.Text != "" {
		mc, err := gollem.NewTextContent(part.Text)
		if err != nil {
			return gollem.MessageContent{}, err
		}
		mc.Meta = meta
		return mc, nil
	}

	// Inline data (image or PDF)
	if part.InlineData != nil {
		if part.InlineData.MIMEType == "application/pdf" {
			return gollem.NewPDFContent(part.InlineData.Data, "")
		}
		return gollem.NewImageContent(
			part.InlineData.MIMEType,
			part.InlineData.Data,
			"",
			"",
		)
	}

	// File data
	if part.FileData != nil {
		// Gemini uses file URIs, store as URL
		return gollem.NewImageContent(
			part.FileData.MIMEType,
			nil,
			part.FileData.FileURI,
			"",
		)
	}

	// Function call
	if part.FunctionCall != nil {
		// Gemini 3.x returns FunctionCall.ID and requires strict id matching on
		// the corresponding FunctionResponse. Older Gemini models leave ID
		// empty, so fall back to a synthesized id that the conversion layer
		// will strip on the way back to Gemini.
		id := part.FunctionCall.ID
		if id == "" {
			id = geminiFallbackToolCallID(part.FunctionCall.Name, fallbackIndex)
		}
		mc, err := gollem.NewToolCallContent(
			id,
			part.FunctionCall.Name,
			part.FunctionCall.Args,
		)
		if err != nil {
			return gollem.MessageContent{}, err
		}
		mc.Meta = meta
		return mc, nil
	}

	// Function response
	if part.FunctionResponse != nil {
		id := part.FunctionResponse.ID
		if id == "" {
			id = geminiFallbackToolCallID(part.FunctionResponse.Name, fallbackIndex)
		}
		return gollem.NewToolResponseContent(
			id,
			part.FunctionResponse.Name,
			part.FunctionResponse.Response,
			false,
		)
	}

	// ThoughtSignature-only part (no text, no function call, not marked as thought)
	// Some Gemini models return parts with only ThoughtSignature set.
	if len(part.ThoughtSignature) > 0 {
		mc, err := gollem.NewTextContent("")
		if err != nil {
			return gollem.MessageContent{}, err
		}
		mc.Meta = meta
		return mc, nil
	}

	return gollem.MessageContent{}, goerr.Wrap(convert.ErrUnsupportedContentType, "unknown Gemini part type")
}

// convertMessagesToGemini converts common Messages to Gemini format
func convertMessagesToGemini(messages []gollem.Message) ([]*genai.Content, error) {
	if len(messages) == 0 {
		return []*genai.Content{}, nil
	}

	// Handle system messages by merging into first user message
	messages = convert.MergeSystemIntoFirstUser(messages)

	// Gemini requires the number of functionResponse parts in a turn to match the number of
	// functionCall parts in the turn it answers, so tool responses split across messages must
	// be sent as one Content.
	// https://ai.google.dev/gemini-api/docs/function-calling ("Parallel function calling")
	// https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/parallel_function_calling.ipynb
	messages = convert.MergeConsecutiveToolMessages(messages)

	result := make([]*genai.Content, 0, len(messages))

	for _, msg := range messages {
		// Skip system messages as they've been merged
		if msg.Role == gollem.RoleSystem {
			continue
		}

		geminiContent, err := convertMessageToGemini(msg)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert message to Gemini format")
		}
		result = append(result, geminiContent)
	}

	return result, nil
}

// convertMessageToGemini converts a single Message to Gemini format
func convertMessageToGemini(msg gollem.Message) (*genai.Content, error) {
	// Convert role
	role := ""
	switch msg.Role {
	case gollem.RoleUser:
		role = "user"
	case gollem.RoleAssistant:
		// Assistant is always converted to model for Gemini SDK
		role = "model"
	case gollem.RoleTool:
		// Tool responses go to user role in Gemini
		role = "user"
	default:
		role = "user"
	}

	// Convert contents
	parts := make([]*genai.Part, 0, len(msg.Contents))
	for _, content := range msg.Contents {
		part, err := convertContentToGemini(content)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert content to Gemini format")
		}
		parts = append(parts, part)
	}

	return &genai.Content{
		Role:  role,
		Parts: parts,
	}, nil
}

// convertContentToGemini converts MessageContent to Gemini part
func convertContentToGemini(content gollem.MessageContent) (*genai.Part, error) {
	// Extract provider metadata if present
	meta, err := unmarshalPartMeta(content.Meta)
	if err != nil {
		return nil, err
	}

	switch content.Type {
	case gollem.MessageContentTypeText:
		textContent, err := content.GetTextContent()
		if err != nil {
			return nil, err
		}
		return &genai.Part{
			Text:             textContent.Text,
			Thought:          meta.Thought,
			ThoughtSignature: meta.ThoughtSignature,
		}, nil

	case gollem.MessageContentTypeThinking:
		reasoningContent, err := content.GetThinkingContent()
		if err != nil {
			return nil, err
		}
		return &genai.Part{
			Text:             reasoningContent.Text,
			Thought:          true,
			ThoughtSignature: meta.ThoughtSignature,
		}, nil

	case gollem.MessageContentTypeImage:
		imgContent, err := content.GetImageContent()
		if err != nil {
			return nil, err
		}
		if len(imgContent.Data) > 0 {
			// Inline data
			return &genai.Part{
				InlineData: &genai.Blob{
					MIMEType: imgContent.MediaType,
					Data:     imgContent.Data,
				},
			}, nil
		} else if imgContent.URL != "" {
			// File URI
			return &genai.Part{
				FileData: &genai.FileData{
					MIMEType: imgContent.MediaType,
					FileURI:  imgContent.URL,
				},
			}, nil
		}
		return nil, goerr.Wrap(convert.ErrInvalidMessageFormat, "image has neither data nor URL")

	case gollem.MessageContentTypePDF:
		pdfContent, err := content.GetPDFContent()
		if err != nil {
			return nil, err
		}
		if len(pdfContent.Data) > 0 {
			return &genai.Part{
				InlineData: &genai.Blob{
					MIMEType: "application/pdf",
					Data:     pdfContent.Data,
				},
			}, nil
		}
		if pdfContent.URL != "" {
			return &genai.Part{
				FileData: &genai.FileData{
					MIMEType: "application/pdf",
					FileURI:  pdfContent.URL,
				},
			}, nil
		}
		return nil, goerr.Wrap(convert.ErrInvalidMessageFormat, "PDF has neither data nor URL")

	case gollem.MessageContentTypeToolCall:
		toolCall, err := content.GetToolCallContent()
		if err != nil {
			return nil, err
		}
		// Skip ids that gollem fabricated internally; they would not match
		// anything Gemini 3.x issued, causing strict-match failures.
		id := toolCall.ID
		if isGeminiFallbackToolCallID(id) {
			id = ""
		}
		return &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   id,
				Name: toolCall.Name,
				Args: toolCall.Arguments,
			},
			ThoughtSignature: meta.ThoughtSignature,
		}, nil

	case gollem.MessageContentTypeToolResponse:
		toolResp, err := content.GetToolResponseContent()
		if err != nil {
			return nil, err
		}
		id := toolResp.ToolCallID
		if isGeminiFallbackToolCallID(id) {
			id = ""
		}
		return &genai.Part{
			FunctionResponse: &genai.FunctionResponse{
				ID:       id,
				Name:     toolResp.Name,
				Response: toolResp.Response,
			},
		}, nil

	default:
		return nil, goerr.Wrap(convert.ErrUnsupportedContentType, "unsupported content type for Gemini", goerr.Value("type", content.Type))
	}
}

// ToContents converts gollem.History to Gemini contents
func ToContents(h *gollem.History) ([]*genai.Content, error) {
	if h == nil || len(h.Messages) == 0 {
		return []*genai.Content{}, nil
	}
	return convertMessagesToGemini(h.Messages)
}

// NewHistory creates gollem.History from Gemini contents
func NewHistory(contents []*genai.Content) (*gollem.History, error) {
	commonMessages, err := convertGeminiToMessages(contents)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to convert Gemini messages to common format")
	}

	return &gollem.History{
		LLType:   gollem.LLMTypeGemini,
		Version:  gollem.HistoryVersion,
		Messages: commonMessages,
	}, nil
}
