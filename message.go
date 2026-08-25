package gollem

import (
	"encoding/json"

	"github.com/gollem-dev/gollem/internal/jsonutil"
)

// Message represents a unified message format that can be converted between different LLM providers.
// All provider-specific messages are converted to this common format for cross-provider compatibility.
type Message struct {
	Role     MessageRole      `json:"role"`
	Contents []MessageContent `json:"contents"`

	// Optional fields for provider-specific information
	Name     string                 `json:"name,omitempty"`     // OpenAI's name field
	Metadata map[string]interface{} `json:"metadata,omitempty"` // Extension metadata
}

// MessageRole represents the role of a message in a conversation
type MessageRole string

const (
	RoleSystem    MessageRole = "system"
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
	RoleTool      MessageRole = "tool" // Tool response (unified across all providers)
)

// MessageContent represents the content of a message in a unified format
type MessageContent struct {
	Type MessageContentType `json:"type"`
	// Data contains type-specific content that should be unmarshaled based on Type
	Data json.RawMessage `json:"data"`
	// Meta contains provider-specific metadata (e.g., Gemini's ThoughtSignature)
	Meta json.RawMessage `json:"meta,omitempty"`
}

// MessageContentType represents the type of content in a message
type MessageContentType string

const (
	MessageContentTypeText         MessageContentType = "text"
	MessageContentTypeImage        MessageContentType = "image"
	MessageContentTypePDF          MessageContentType = "pdf"
	MessageContentTypeToolCall     MessageContentType = "tool_call"
	MessageContentTypeToolResponse MessageContentType = "tool_response"
	MessageContentTypeThinking     MessageContentType = "thinking"
)

// TextContent represents text content in a message
type TextContent struct {
	Text string `json:"text"`
}

// ThinkingContent represents thinking/reasoning content
type ThinkingContent struct {
	Text string `json:"text"`
}

// ImageContent represents image content in a message
type ImageContent struct {
	MediaType string `json:"media_type,omitempty"` // e.g., "image/jpeg", "image/png"
	Data      []byte `json:"data,omitempty"`       // Image data (base64 encoded in JSON)
	URL       string `json:"url,omitempty"`        // Image URL (either Data or URL should be set)
	Detail    string `json:"detail,omitempty"`     // OpenAI: "high", "low", "auto"
}

// PDFContent represents PDF document content in a message
type PDFContent struct {
	Data []byte `json:"data,omitempty"` // PDF data (base64 encoded in JSON)
	URL  string `json:"url,omitempty"`  // PDF URL (for future URL source support)
}

// ToolCallContent represents a tool/function call request
type ToolCallContent struct {
	ID        string                 `json:"id"`        // Call ID for matching with response
	Name      string                 `json:"name"`      // Tool/function name
	Arguments map[string]interface{} `json:"arguments"` // Arguments as JSON object
}

// ToolResponseContent represents a tool/function response
type ToolResponseContent struct {
	ToolCallID string                 `json:"tool_call_id"`       // ID of the corresponding call
	Name       string                 `json:"name,omitempty"`     // Tool/function name (required for Gemini)
	Response   map[string]interface{} `json:"response"`           // Response content
	IsError    bool                   `json:"is_error,omitempty"` // Whether this is an error response (Claude)
}

// makeContent marshals v into JSON and wraps it in a MessageContent with the given type.
func makeContent[T any](t MessageContentType, v T) (MessageContent, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return MessageContent{}, err
	}
	return MessageContent{Type: t, Data: data}, nil
}

// decodeContent checks that mc has the expected type, then decodes its Data into T.
// The decode preserves numbers that a float64 cannot represent exactly, so a tool
// argument or result stored in a History is replayed to the provider unchanged.
func decodeContent[T any](t MessageContentType, mc *MessageContent) (*T, error) {
	if mc.Type != t {
		return nil, ErrInvalidHistoryData
	}
	var content T
	if err := jsonutil.Decode(mc.Data, &content); err != nil {
		return nil, err
	}
	return &content, nil
}

// Helper methods for creating MessageContent

// NewTextContent creates a new text message content
func NewTextContent(text string) (MessageContent, error) {
	return makeContent(MessageContentTypeText, TextContent{Text: text})
}

// NewThinkingContent creates a new thinking message content
func NewThinkingContent(text string) (MessageContent, error) {
	return makeContent(MessageContentTypeThinking, ThinkingContent{Text: text})
}

// NewImageContent creates a new image message content
func NewImageContent(mediaType string, imageData []byte, url string, detail string) (MessageContent, error) {
	return makeContent(MessageContentTypeImage, ImageContent{
		MediaType: mediaType,
		Data:      imageData,
		URL:       url,
		Detail:    detail,
	})
}

// NewPDFContent creates a new PDF message content
func NewPDFContent(pdfData []byte, url string) (MessageContent, error) {
	return makeContent(MessageContentTypePDF, PDFContent{Data: pdfData, URL: url})
}

// NewToolCallContent creates a new tool call message content
func NewToolCallContent(id, name string, args map[string]interface{}) (MessageContent, error) {
	return makeContent(MessageContentTypeToolCall, ToolCallContent{
		ID:        id,
		Name:      name,
		Arguments: args,
	})
}

// NewToolResponseContent creates a new tool response message content
func NewToolResponseContent(toolCallID, name string, response map[string]interface{}, isError bool) (MessageContent, error) {
	return makeContent(MessageContentTypeToolResponse, ToolResponseContent{
		ToolCallID: toolCallID,
		Name:       name,
		Response:   response,
		IsError:    isError,
	})
}

// Helper methods for extracting content from MessageContent

// GetTextContent extracts text content from a MessageContent
func (mc *MessageContent) GetTextContent() (*TextContent, error) {
	return decodeContent[TextContent](MessageContentTypeText, mc)
}

// GetImageContent extracts image content from a MessageContent
func (mc *MessageContent) GetImageContent() (*ImageContent, error) {
	return decodeContent[ImageContent](MessageContentTypeImage, mc)
}

// GetPDFContent extracts PDF content from a MessageContent
func (mc *MessageContent) GetPDFContent() (*PDFContent, error) {
	return decodeContent[PDFContent](MessageContentTypePDF, mc)
}

// GetToolCallContent extracts tool call content from a MessageContent
func (mc *MessageContent) GetToolCallContent() (*ToolCallContent, error) {
	return decodeContent[ToolCallContent](MessageContentTypeToolCall, mc)
}

// GetToolResponseContent extracts tool response content from a MessageContent
func (mc *MessageContent) GetToolResponseContent() (*ToolResponseContent, error) {
	return decodeContent[ToolResponseContent](MessageContentTypeToolResponse, mc)
}

// GetThinkingContent extracts thinking content from a MessageContent
func (mc *MessageContent) GetThinkingContent() (*ThinkingContent, error) {
	return decodeContent[ThinkingContent](MessageContentTypeThinking, mc)
}
