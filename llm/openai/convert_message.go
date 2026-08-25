package openai

import (
	"encoding/base64"
	"strings"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/internal/convert"
	"github.com/gollem-dev/gollem/internal/jsonutil"
	"github.com/m-mizutani/goerr/v2"
	"github.com/sashabaranov/go-openai"
)

const (
	rawArgumentsKey = "arguments"
	rawContentKey   = "content"
)

// convertOpenAIToMessages converts OpenAI messages to common Message format
func convertOpenAIToMessages(messages []openai.ChatCompletionMessage) ([]gollem.Message, error) {
	if len(messages) == 0 {
		return []gollem.Message{}, nil
	}

	result := make([]gollem.Message, 0, len(messages))

	for _, msg := range messages {
		commonMsg, err := convertOpenAIMessage(msg)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert OpenAI message")
		}
		result = append(result, commonMsg)
	}

	return result, nil
}

// convertOpenAIMessage converts a single OpenAI message to common format
func convertOpenAIMessage(msg openai.ChatCompletionMessage) (gollem.Message, error) {
	contents := make([]gollem.MessageContent, 0)

	// Handle different content types
	// Handle reasoning content first (it should come before text content)
	if msg.ReasoningContent != "" {
		content, err := gollem.NewThinkingContent(msg.ReasoningContent)
		if err != nil {
			return gollem.Message{}, err
		}
		contents = append(contents, content)
	}

	// Skip text content for tool/function roles as they have special handling
	if msg.Content != "" && msg.Role != "tool" && msg.Role != "function" {
		// Simple text content
		content, err := gollem.NewTextContent(msg.Content)
		if err != nil {
			return gollem.Message{}, err
		}
		contents = append(contents, content)
	}

	// Handle MultiContent (for messages with text and images)
	if msg.MultiContent != nil {
		for _, part := range msg.MultiContent {
			if part.Type == "text" {
				content, err := gollem.NewTextContent(part.Text)
				if err != nil {
					return gollem.Message{}, err
				}
				contents = append(contents, content)
			} else if part.Type == "image_url" && part.ImageURL != nil {
				// Extract data from URL if it's a data URL
				url := part.ImageURL.URL
				detail := string(part.ImageURL.Detail)

				// Parse data URLs to extract base64 data
				if len(url) > 5 && url[:5] == "data:" {
					if idx := strings.Index(url, ";base64,"); idx != -1 {
						mediaType := url[5:idx]
						base64Data := url[idx+8:]

						// Check if this is a PDF data URL
						if mediaType == "application/pdf" {
							pdfData, err := base64.StdEncoding.DecodeString(base64Data)
							if err != nil {
								return gollem.Message{}, goerr.Wrap(err, "failed to decode base64 PDF data from data URL")
							}
							mc, err := gollem.NewPDFContent(pdfData, "")
							if err != nil {
								return gollem.Message{}, err
							}
							contents = append(contents, mc)
							continue
						}

						// Image data URL
						imageData, err := base64.StdEncoding.DecodeString(base64Data)
						if err != nil {
							return gollem.Message{}, goerr.Wrap(err, "failed to decode base64 image data from data URL")
						}
						url = ""
						content, err := gollem.NewImageContent(mediaType, imageData, url, detail)
						if err != nil {
							return gollem.Message{}, err
						}
						contents = append(contents, content)
						continue
					}
				}

				// Non-data URL image
				content, err := gollem.NewImageContent("", nil, url, detail)
				if err != nil {
					return gollem.Message{}, err
				}
				contents = append(contents, content)
			}
		}
	}

	// Handle tool calls
	if msg.ToolCalls != nil {
		for _, toolCall := range msg.ToolCalls {
			if toolCall.Function.Name != "" {
				args, err := convert.ParseJSONArguments(toolCall.Function.Arguments)
				if err != nil {
					// Use raw string if parsing fails
					args = map[string]interface{}{rawArgumentsKey: toolCall.Function.Arguments}
				}
				content, err := gollem.NewToolCallContent(toolCall.ID, toolCall.Function.Name, args)
				if err != nil {
					return gollem.Message{}, err
				}
				contents = append(contents, content)
			}
		}
	}

	// Handle legacy function call - convert to tool call
	if msg.FunctionCall != nil {
		args, err := convert.ParseJSONArguments(msg.FunctionCall.Arguments)
		if err != nil {
			// Use raw string if parsing fails
			args = map[string]interface{}{rawArgumentsKey: msg.FunctionCall.Arguments}
		}
		// Generate tool call ID for legacy function calls
		toolCallID := convert.GenerateToolCallID(msg.FunctionCall.Name, 0)
		content, err := gollem.NewToolCallContent(toolCallID, msg.FunctionCall.Name, args)
		if err != nil {
			return gollem.Message{}, err
		}
		contents = append(contents, content)
	}

	// Handle tool responses (tool role messages)
	if msg.Role == "tool" && msg.ToolCallID != "" {
		// Parse content as response
		response, err := jsonutil.DecodeObject([]byte(msg.Content))
		if err != nil {
			// If not JSON, wrap in a response object
			response = map[string]interface{}{rawContentKey: msg.Content}
		}
		content, err := gollem.NewToolResponseContent(msg.ToolCallID, msg.Name, response, false)
		if err != nil {
			return gollem.Message{}, err
		}
		contents = append(contents, content)
	}

	// Handle legacy function responses - convert to tool response
	if msg.Role == "function" {
		// Parse content as response
		response, err := jsonutil.DecodeObject([]byte(msg.Content))
		if err != nil {
			// If not JSON, wrap in a response object
			response = map[string]interface{}{rawContentKey: msg.Content}
		}
		// Generate tool call ID for legacy function responses
		toolCallID := convert.GenerateToolCallID(msg.Name, 0)
		content, err := gollem.NewToolResponseContent(toolCallID, msg.Name, response, false)
		if err != nil {
			return gollem.Message{}, err
		}
		contents = append(contents, content)
	}

	return gollem.Message{
		Role:     convert.ConvertRoleToCommon(msg.Role),
		Contents: contents,
		Name:     msg.Name,
	}, nil
}

// convertMessagesToOpenAI converts common Messages to OpenAI format
func convertMessagesToOpenAI(messages []gollem.Message) ([]openai.ChatCompletionMessage, error) {
	if len(messages) == 0 {
		return []openai.ChatCompletionMessage{}, nil
	}

	result := make([]openai.ChatCompletionMessage, 0, len(messages))

	for _, msg := range messages {
		openaiMsgs, err := convertMessageToOpenAI(msg)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to convert message to OpenAI format")
		}
		result = append(result, openaiMsgs...)
	}

	return result, nil
}

// convertMessageToOpenAI converts a single Message to OpenAI format
// Returns multiple messages as tool responses need separate messages
func convertMessageToOpenAI(msg gollem.Message) ([]openai.ChatCompletionMessage, error) {
	// Handle role conversion
	role := ""
	switch msg.Role {
	case gollem.RoleSystem:
		role = "system"
	case gollem.RoleUser:
		role = "user"
	case gollem.RoleAssistant:
		role = "assistant"
	case gollem.RoleTool:
		role = "tool"
	default:
		role = string(msg.Role)
	}

	// Check if this is a simple text message
	if len(msg.Contents) == 1 && msg.Contents[0].Type == gollem.MessageContentTypeText {
		textContent, err := msg.Contents[0].GetTextContent()
		if err != nil {
			return nil, err
		}
		return []openai.ChatCompletionMessage{{
			Role:    role,
			Content: textContent.Text,
			Name:    msg.Name,
		}}, nil
	}

	// Handle complex messages
	var textParts []openai.ChatMessagePart
	var toolCalls []openai.ToolCall
	var toolResponses []openai.ChatCompletionMessage
	var reasoningContent string

	for _, content := range msg.Contents {
		switch content.Type {
		case gollem.MessageContentTypeThinking:
			thinkingContent, err := content.GetThinkingContent()
			if err != nil {
				return nil, goerr.Wrap(err, "failed to get thinking content")
			}
			// Accumulate reasoning content (will be set on the main message)
			reasoningContent += thinkingContent.Text

		case gollem.MessageContentTypeText:
			textContent, err := content.GetTextContent()
			if err != nil {
				return nil, goerr.Wrap(err, "failed to get text content")
			}
			textParts = append(textParts, openai.ChatMessagePart{
				Type: "text",
				Text: textContent.Text,
			})

		case gollem.MessageContentTypeImage:
			imgContent, err := content.GetImageContent()
			if err != nil {
				return nil, goerr.Wrap(err, "failed to get image content")
			}
			imageURL := imgContent.URL
			if len(imgContent.Data) > 0 {
				// Convert to data URL with Base64 encoding
				imageURL = "data:" + imgContent.MediaType + ";base64," + base64.StdEncoding.EncodeToString(imgContent.Data)
			}
			textParts = append(textParts, openai.ChatMessagePart{
				Type: "image_url",
				ImageURL: &openai.ChatMessageImageURL{
					URL:    imageURL,
					Detail: openai.ImageURLDetail(imgContent.Detail),
				},
			})

		case gollem.MessageContentTypePDF:
			pdfContent, err := content.GetPDFContent()
			if err != nil {
				return nil, goerr.Wrap(err, "failed to get PDF content")
			}
			// Use data URL in image_url field (OpenAI SDK workaround)
			pdfURL := ""
			if len(pdfContent.Data) > 0 {
				pdfURL = "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(pdfContent.Data)
			} else if pdfContent.URL != "" {
				pdfURL = pdfContent.URL
			}
			if pdfURL != "" {
				textParts = append(textParts, openai.ChatMessagePart{
					Type: "image_url",
					ImageURL: &openai.ChatMessageImageURL{
						URL: pdfURL,
					},
				})
			}

		case gollem.MessageContentTypeToolCall:
			toolCall, err := content.GetToolCallContent()
			if err != nil {
				return nil, goerr.Wrap(err, "failed to get tool call content")
			}
			args, err := convert.StringifyJSONArguments(toolCall.Arguments)
			if err != nil {
				args = "{}"
			}
			toolCalls = append(toolCalls, openai.ToolCall{
				ID:   toolCall.ID,
				Type: "function",
				Function: openai.FunctionCall{
					Name:      toolCall.Name,
					Arguments: args,
				},
			})

		case gollem.MessageContentTypeToolResponse:
			toolResp, err := content.GetToolResponseContent()
			if err != nil {
				return nil, goerr.Wrap(err, "failed to get tool response content")
			}
			respStr, err := convert.StringifyJSONArguments(toolResp.Response)
			if err != nil {
				respStr = "{}"
			}
			toolResponses = append(toolResponses, openai.ChatCompletionMessage{
				Role:       "tool",
				Content:    respStr,
				Name:       toolResp.Name,
				ToolCallID: toolResp.ToolCallID,
			})
		}
	}

	// Build the result messages
	var result []openai.ChatCompletionMessage

	// Special handling for tool role - merge text content with tool response
	if role == "tool" && len(toolResponses) > 0 {
		// For tool messages, use the tool response as the main message and merge text content
		for _, toolResp := range toolResponses {
			if len(textParts) == 1 && textParts[0].Type == "text" {
				// Replace content with the text content if we have it
				toolResp.Content = textParts[0].Text
			}
			result = append(result, toolResp)
		}
	} else {
		// Add main message if it has content or tool calls
		if len(textParts) > 0 || len(toolCalls) > 0 {
			mainMsg := openai.ChatCompletionMessage{
				Role: role,
				Name: msg.Name,
			}

			// Set content based on what we have
			if len(textParts) == 1 && textParts[0].Type == "text" {
				// Simple text content
				mainMsg.Content = textParts[0].Text
			} else if len(textParts) > 0 {
				// Multi-part content
				mainMsg.MultiContent = textParts
			}

			// Add tool calls
			if len(toolCalls) > 0 {
				mainMsg.ToolCalls = toolCalls
			}

			// Add reasoning content if present
			if reasoningContent != "" {
				mainMsg.ReasoningContent = reasoningContent
			}

			result = append(result, mainMsg)
		}

		// Add tool response messages (they must be separate messages for non-tool roles)
		result = append(result, toolResponses...)
	}

	// If we have no messages, create an empty text message
	if len(result) == 0 {
		result = append(result, openai.ChatCompletionMessage{
			Role:    role,
			Content: "",
			Name:    msg.Name,
		})
	}

	return result, nil
}

// ToMessages converts gollem.History to OpenAI messages
func ToMessages(h *gollem.History) ([]openai.ChatCompletionMessage, error) {
	if h == nil || len(h.Messages) == 0 {
		return []openai.ChatCompletionMessage{}, nil
	}
	return convertMessagesToOpenAI(h.Messages)
}

// NewHistory creates gollem.History from OpenAI messages
func NewHistory(messages []openai.ChatCompletionMessage) (*gollem.History, error) {
	commonMessages, err := convertOpenAIToMessages(messages)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to convert OpenAI messages to common format")
	}

	return &gollem.History{
		LLType:   gollem.LLMTypeOpenAI,
		Version:  gollem.HistoryVersion,
		Messages: commonMessages,
	}, nil
}
