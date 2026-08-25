package claude

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/internal/convert"
	"github.com/gollem-dev/gollem/internal/jsonutil"
	"github.com/m-mizutani/goerr/v2"
)

// claudePartMeta is the metadata stored in MessageContent.Meta for Claude content blocks.
// It preserves Claude-specific fields (e.g., thinking signatures, redacted status) across
// serialization/deserialization without polluting the common message types.
type claudePartMeta struct {
	Signature string `json:"signature,omitempty"` // Signature for thinking blocks
	Redacted  bool   `json:"redacted,omitempty"`  // Whether this is a redacted thinking block
}

// marshalClaudePartMeta marshals claudePartMeta to JSON for MessageContent.Meta.
// Returns nil if no metadata needs to be stored.
func marshalClaudePartMeta(m claudePartMeta) (json.RawMessage, error) {
	if !m.Redacted && m.Signature == "" {
		return nil, nil
	}
	data, err := json.Marshal(m)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to marshal claude part meta")
	}
	return data, nil
}

// unmarshalClaudePartMeta unmarshals claudePartMeta from MessageContent.Meta.
// Returns zero-value claudePartMeta if meta is nil or empty.
func unmarshalClaudePartMeta(meta json.RawMessage) (claudePartMeta, error) {
	if len(meta) == 0 {
		return claudePartMeta{}, nil
	}
	var m claudePartMeta
	if err := json.Unmarshal(meta, &m); err != nil {
		return claudePartMeta{}, goerr.Wrap(err, "failed to unmarshal claude part meta")
	}
	return m, nil
}

// convertClaudeToMessages converts Claude messages to common Message format
func convertClaudeToMessages(messages []anthropic.MessageParam) ([]gollem.Message, error) {
	if len(messages) == 0 {
		return []gollem.Message{}, nil
	}

	toolNamesByID := collectClaudeToolNames(messages)

	result := make([]gollem.Message, 0, len(messages))

	for _, msg := range messages {
		contents := make([]gollem.MessageContent, 0, len(msg.Content))

		for _, block := range msg.Content {
			content, err := convertClaudeContentBlock(block, toolNamesByID)
			if err != nil {
				// Skip unsupported content types (like empty text blocks)
				if err == convert.ErrUnsupportedContentType {
					continue
				}
				return nil, goerr.Wrap(err, "failed to convert Claude content block")
			}
			contents = append(contents, content)
		}

		// Skip messages with no content after conversion
		if len(contents) == 0 {
			continue
		}

		result = append(result, gollem.Message{
			Role:     convert.ConvertRoleToCommon(string(msg.Role)),
			Contents: contents,
		})
	}

	return result, nil
}

// toolUseInput encodes tool call arguments for anthropic.ToolUseBlockParam.Input.
//
// The arguments are encoded here rather than handed to the SDK as a map: the Anthropic
// SDK's own JSON encoder writes a json.Number as a quoted string, which would change an
// argument's type on the wire. A json.RawMessage implements json.Marshaler, so the SDK
// emits these bytes verbatim. Every site that builds a tool_use block must use this, or
// that site sends a wide integer argument as a string.
func toolUseInput(args map[string]any) (json.RawMessage, error) {
	raw, err := json.Marshal(args)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(raw), nil
}

// collectClaudeToolNames maps each tool_use ID to the tool name it called. Claude's
// tool_result blocks reference a tool_use only by ID, so the name has to be recovered from
// the whole message list rather than from the block being converted.
func collectClaudeToolNames(messages []anthropic.MessageParam) map[string]string {
	names := make(map[string]string)
	for _, msg := range messages {
		for _, block := range msg.Content {
			if block.OfToolUse != nil && block.OfToolUse.ID != "" {
				names[block.OfToolUse.ID] = block.OfToolUse.Name
			}
		}
	}
	return names
}

// convertClaudeContentBlock converts a single Claude content block to MessageContent.
// toolNamesByID supplies the tool name for tool_result blocks, which do not carry one.
func convertClaudeContentBlock(block anthropic.ContentBlockParamUnion, toolNamesByID map[string]string) (gollem.MessageContent, error) {
	// Handle text blocks
	if block.OfText != nil {
		// Skip empty text blocks
		if block.OfText.Text == "" {
			return gollem.MessageContent{}, convert.ErrUnsupportedContentType
		}
		return gollem.NewTextContent(block.OfText.Text)
	}

	// Handle thinking blocks
	if block.OfThinking != nil {
		mc, err := gollem.NewThinkingContent(block.OfThinking.Thinking)
		if err != nil {
			return gollem.MessageContent{}, err
		}
		// Store signature in meta for multi-turn conversations
		if block.OfThinking.Signature != "" {
			meta, err := marshalClaudePartMeta(claudePartMeta{
				Signature: block.OfThinking.Signature,
			})
			if err != nil {
				return gollem.MessageContent{}, err
			}
			mc.Meta = meta
		}
		return mc, nil
	}

	// Handle redacted thinking blocks
	if block.OfRedactedThinking != nil {
		mc, err := gollem.NewThinkingContent("") // Empty text for redacted blocks
		if err != nil {
			return gollem.MessageContent{}, err
		}
		// Store signature and redacted flag in meta
		meta, err := marshalClaudePartMeta(claudePartMeta{
			Signature: block.OfRedactedThinking.Data,
			Redacted:  true,
		})
		if err != nil {
			return gollem.MessageContent{}, err
		}
		mc.Meta = meta
		return mc, nil
	}

	// Handle image blocks
	if block.OfImage != nil {
		if block.OfImage.Source.OfBase64 != nil {
			// Decode the Base64 string to raw bytes
			decodedData, err := base64.StdEncoding.DecodeString(block.OfImage.Source.OfBase64.Data)
			if err != nil {
				return gollem.MessageContent{}, goerr.Wrap(err, "failed to decode base64 image data",
					goerr.V("media_type", string(block.OfImage.Source.OfBase64.MediaType)))
			}
			return gollem.NewImageContent(
				string(block.OfImage.Source.OfBase64.MediaType),
				decodedData,
				"",
				"",
			)
		}
		// Handle URL images if supported
		// Note: Claude API primarily uses base64 images
	}

	// Handle document blocks (PDF)
	if block.OfDocument != nil {
		if block.OfDocument.Source.OfBase64 != nil {
			decodedData, err := base64.StdEncoding.DecodeString(block.OfDocument.Source.OfBase64.Data)
			if err != nil {
				return gollem.MessageContent{}, goerr.Wrap(err, "failed to decode base64 PDF data")
			}
			return gollem.NewPDFContent(decodedData, "")
		}
		if block.OfDocument.Source.OfURL != nil {
			return gollem.NewPDFContent(nil, block.OfDocument.Source.OfURL.URL)
		}
	}

	// Handle tool use blocks
	if block.OfToolUse != nil {
		// Convert input to map if it's not already
		var args map[string]interface{}
		switch v := block.OfToolUse.Input.(type) {
		case map[string]interface{}:
			args = v
		case json.RawMessage:
			// The shape convertContentToClaude produces.
			parsed, err := jsonutil.DecodeObject(v)
			if err != nil {
				return gollem.MessageContent{}, goerr.Wrap(err, "failed to decode tool use input",
					goerr.V("tool", block.OfToolUse.Name))
			}
			args = parsed
		case string:
			// A tool_use input arriving as a string is the raw arguments JSON. When it does
			// not parse, the string itself is the only thing left to carry, so keep it under
			// "input" rather than losing the arguments entirely.
			parsed, err := jsonutil.DecodeObject([]byte(v))
			if err != nil {
				args = map[string]interface{}{"input": v}
			} else {
				args = parsed
			}
		default:
			// Convert to JSON then back to map
			data, err := json.Marshal(v)
			if err != nil {
				return gollem.MessageContent{}, goerr.Wrap(err, "failed to encode tool use input",
					goerr.V("tool", block.OfToolUse.Name))
			}
			args, err = jsonutil.DecodeObject(data)
			if err != nil {
				return gollem.MessageContent{}, goerr.Wrap(err, "failed to decode tool use input",
					goerr.V("tool", block.OfToolUse.Name))
			}
		}

		return gollem.NewToolCallContent(
			block.OfToolUse.ID,
			block.OfToolUse.Name,
			args,
		)
	}

	// Handle tool result blocks
	if block.OfToolResult != nil {
		// Extract text content from tool result
		responseText := ""
		if len(block.OfToolResult.Content) > 0 && block.OfToolResult.Content[0].OfText != nil {
			responseText = block.OfToolResult.Content[0].OfText.Text
		}

		isError := false
		if block.OfToolResult.IsError.Valid() {
			isError = block.OfToolResult.IsError.Value
		}

		// Try to parse responseText as JSON to preserve structure
		response, err := jsonutil.DecodeObject([]byte(responseText))
		if err != nil {
			// If not valid JSON, wrap in content field
			response = map[string]interface{}{"content": responseText}
		}

		// A Claude tool_result block carries no tool name, but Gemini requires one on every
		// functionResponse part. Recover it from the tool_use block with the same ID so a
		// History produced by Claude can be replayed against another provider.
		return gollem.NewToolResponseContent(
			block.OfToolResult.ToolUseID,
			toolNamesByID[block.OfToolResult.ToolUseID],
			response,
			isError,
		)
	}

	return gollem.MessageContent{}, goerr.Wrap(convert.ErrUnsupportedContentType, "unknown Claude content block type")
}

// convertMessagesToClaude converts common Messages to Claude format
func convertMessagesToClaude(messages []gollem.Message) ([]anthropic.MessageParam, error) {
	if len(messages) == 0 {
		return []anthropic.MessageParam{}, nil
	}

	// Handle system messages by merging into first user message
	messages, err := convert.MergeSystemIntoFirstUser(messages)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to merge system message into first user message")
	}

	// Claude requires one tool_result for each tool_use block, all together in the next user
	// message, so tool responses split across messages must be sent as one message.
	// https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use
	messages = convert.MergeConsecutiveToolMessages(messages)

	result := make([]anthropic.MessageParam, 0, len(messages))

	for _, msg := range messages {
		// Skip system messages as they've been merged
		if msg.Role == gollem.RoleSystem {
			continue
		}
		// Skip empty messages
		if len(msg.Contents) == 0 {
			continue
		}

		claudeMsg, err := convertMessageToClaude(msg)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert message to Claude format")
		}
		// Skip messages with no content after conversion
		if len(claudeMsg.Content) == 0 {
			continue
		}
		result = append(result, claudeMsg)
	}

	return result, nil
}

// convertMessageToClaude converts a single Message to Claude format
func convertMessageToClaude(msg gollem.Message) (anthropic.MessageParam, error) {
	// Convert role
	var role anthropic.MessageParamRole
	switch msg.Role {
	case gollem.RoleUser:
		role = anthropic.MessageParamRoleUser
	case gollem.RoleAssistant:
		role = anthropic.MessageParamRoleAssistant
	case gollem.RoleTool:
		// Tool responses should be in user role with tool_result block
		role = anthropic.MessageParamRoleUser
	default:
		role = anthropic.MessageParamRoleUser
	}

	// Convert contents
	contents := make([]anthropic.ContentBlockParamUnion, 0, len(msg.Contents))
	for _, content := range msg.Contents {
		claudeContent, err := convertContentToClaude(content, msg.Role)
		if err != nil {
			// Skip unsupported content types instead of failing completely
			if err == convert.ErrUnsupportedContentType {
				continue
			}
			return anthropic.MessageParam{}, goerr.Wrap(err, "failed to convert content to Claude format")
		}
		contents = append(contents, claudeContent)
	}

	return anthropic.MessageParam{
		Role:    role,
		Content: contents,
	}, nil
}

// convertContentToClaude converts MessageContent to Claude content block
func convertContentToClaude(content gollem.MessageContent, messageRole gollem.MessageRole) (anthropic.ContentBlockParamUnion, error) {
	_ = messageRole // Currently unused but may be needed for future conversions
	switch content.Type {
	case gollem.MessageContentTypeText:
		textContent, err := content.GetTextContent()
		if err != nil {
			return anthropic.ContentBlockParamUnion{}, err
		}
		// Skip empty text content
		if textContent.Text == "" {
			return anthropic.ContentBlockParamUnion{}, convert.ErrUnsupportedContentType
		}
		return anthropic.NewTextBlock(textContent.Text), nil

	case gollem.MessageContentTypeThinking:
		thinkingContent, err := content.GetThinkingContent()
		if err != nil {
			return anthropic.ContentBlockParamUnion{}, err
		}

		// Extract metadata to determine if this is a redacted block
		meta, err := unmarshalClaudePartMeta(content.Meta)
		if err != nil {
			return anthropic.ContentBlockParamUnion{}, err
		}

		// Handle redacted thinking blocks
		if meta.Redacted {
			return anthropic.NewRedactedThinkingBlock(meta.Signature), nil
		}

		// Handle normal thinking blocks with signature
		return anthropic.NewThinkingBlock(meta.Signature, thinkingContent.Text), nil

	case gollem.MessageContentTypeImage:
		imgContent, err := content.GetImageContent()
		if err != nil {
			return anthropic.ContentBlockParamUnion{}, err
		}
		// Convert to base64 if we have raw data
		if len(imgContent.Data) > 0 {
			return anthropic.NewImageBlockBase64(imgContent.MediaType, base64.StdEncoding.EncodeToString(imgContent.Data)), nil
		}
		// For URL images, create a text block with the URL reference
		// This maintains the information even though Claude can't directly display the image
		if imgContent.URL != "" {
			imageRef := fmt.Sprintf("[Image: %s]", imgContent.URL)
			if imgContent.Detail != "" {
				imageRef = fmt.Sprintf("[Image (%s): %s]", imgContent.Detail, imgContent.URL)
			}
			return anthropic.NewTextBlock(imageRef), nil
		}
		return anthropic.ContentBlockParamUnion{}, convert.ErrUnsupportedContentType

	case gollem.MessageContentTypePDF:
		pdfContent, err := content.GetPDFContent()
		if err != nil {
			return anthropic.ContentBlockParamUnion{}, err
		}
		if len(pdfContent.Data) > 0 {
			return anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
				Data: base64.StdEncoding.EncodeToString(pdfContent.Data),
			}), nil
		}
		if pdfContent.URL != "" {
			return anthropic.NewDocumentBlock(anthropic.URLPDFSourceParam{
				URL: pdfContent.URL,
			}), nil
		}
		return anthropic.ContentBlockParamUnion{}, convert.ErrUnsupportedContentType

	case gollem.MessageContentTypeToolCall:
		toolCall, err := content.GetToolCallContent()
		if err != nil {
			return anthropic.ContentBlockParamUnion{}, err
		}
		input, err := toolUseInput(toolCall.Arguments)
		if err != nil {
			return anthropic.ContentBlockParamUnion{}, goerr.Wrap(err, "failed to encode tool call arguments",
				goerr.V("tool", toolCall.Name))
		}
		return anthropic.NewToolUseBlock(toolCall.ID, input, toolCall.Name), nil

	case gollem.MessageContentTypeToolResponse:
		toolResp, err := content.GetToolResponseContent()
		if err != nil {
			return anthropic.ContentBlockParamUnion{}, err
		}
		// Extract content string from response map
		contentStr := ""
		if c, ok := toolResp.Response["content"].(string); ok {
			contentStr = c
		} else {
			// Try to JSON stringify the response
			data, err := json.Marshal(toolResp.Response)
			if err != nil {
				return anthropic.ContentBlockParamUnion{}, goerr.Wrap(err, "failed to encode tool response",
					goerr.V("tool_call_id", toolResp.ToolCallID))
			}
			contentStr = string(data)
		}

		return anthropic.NewToolResultBlock(toolResp.ToolCallID, contentStr, toolResp.IsError), nil

	default:
		return anthropic.ContentBlockParamUnion{}, goerr.Wrap(convert.ErrUnsupportedContentType, "unsupported content type for Claude", goerr.Value("type", content.Type))
	}
}

// ToMessages converts gollem.History to Claude messages
func ToMessages(h *gollem.History) ([]anthropic.MessageParam, error) {
	if h == nil || len(h.Messages) == 0 {
		return []anthropic.MessageParam{}, nil
	}
	return convertMessagesToClaude(h.Messages)
}

// NewHistory creates gollem.History from Claude messages
func NewHistory(messages []anthropic.MessageParam) (*gollem.History, error) {
	commonMessages, err := convertClaudeToMessages(messages)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to convert Claude messages to common format")
	}

	return &gollem.History{
		LLType:   gollem.LLMTypeClaude,
		Version:  gollem.HistoryVersion,
		Messages: commonMessages,
	}, nil
}
