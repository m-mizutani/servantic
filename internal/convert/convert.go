package convert

import (
	"encoding/json"
	"errors"
	"strconv"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

// Conversion errors
var (
	// ErrUnsupportedContentType is returned when a content type cannot be converted
	ErrUnsupportedContentType = errors.New("unsupported content type")

	// ErrInvalidMessageFormat is returned when a message has an invalid format
	ErrInvalidMessageFormat = errors.New("invalid message format")

	// ErrConversionFailed is returned when conversion between formats fails
	ErrConversionFailed = errors.New("conversion failed")
)

// ConversionWarning represents a warning during conversion
type ConversionWarning struct {
	Message string
	Field   string
	Value   interface{}
}

// ParseJSONArguments attempts to parse a JSON string into a map
func ParseJSONArguments(jsonStr string) (map[string]interface{}, error) {
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &args); err != nil {
		return nil, goerr.Wrap(err, "failed to parse JSON arguments")
	}
	return args, nil
}

// StringifyJSONArguments converts a map to a JSON string
func StringifyJSONArguments(args map[string]interface{}) (string, error) {
	data, err := json.Marshal(args)
	if err != nil {
		return "", goerr.Wrap(err, "failed to stringify JSON arguments")
	}
	return string(data), nil
}

// ConvertRoleToCommon converts various provider roles to common MessageRole
func ConvertRoleToCommon(role string) gollem.MessageRole {
	switch role {
	case "system":
		return gollem.RoleSystem
	case "user":
		return gollem.RoleUser
	case "assistant":
		return gollem.RoleAssistant
	case "tool":
		return gollem.RoleTool
	case "function":
		// Legacy OpenAI function role is treated as tool
		return gollem.RoleTool
	case "model":
		// Gemini's model role is treated as assistant
		return gollem.RoleAssistant
	default:
		// Default to user role for unknown roles
		return gollem.RoleUser
	}
}

// MergeSystemIntoFirstUser merges a system message into the first user message
// This is used for providers that don't support system messages directly (Claude, Gemini)
//
// The given slice and its messages are left untouched: the caller's History must survive a
// conversion unchanged, since the same History is converted again on every later request.
func MergeSystemIntoFirstUser(messages []gollem.Message) []gollem.Message {
	if len(messages) == 0 {
		return messages
	}

	// Find the first system message
	var systemContent string
	systemIndex := -1
	for i, msg := range messages {
		if msg.Role == gollem.RoleSystem {
			systemIndex = i
			// Extract text content from system message
			for _, content := range msg.Contents {
				if content.Type == gollem.MessageContentTypeText {
					var textContent gollem.TextContent
					if err := json.Unmarshal(content.Data, &textContent); err == nil {
						if systemContent != "" {
							systemContent += "\n"
						}
						systemContent += textContent.Text
					}
				}
			}
			break
		}
	}

	if systemIndex < 0 {
		return messages
	}

	// Remove the system message from the list
	result := make([]gollem.Message, 0, len(messages)-1)
	result = append(result, messages[:systemIndex]...)
	result = append(result, messages[systemIndex+1:]...)

	if systemContent == "" {
		return result
	}

	// Find first user message and prepend system content
	for i, msg := range result {
		if msg.Role == gollem.RoleUser {
			// Prepend system content to first user message
			newContent := make([]gollem.MessageContent, 0, len(msg.Contents)+1)

			// Add system content first
			if textContent, err := gollem.NewTextContent(systemContent + "\n\n"); err == nil {
				newContent = append(newContent, textContent)
			}

			// Add existing user content
			newContent = append(newContent, msg.Contents...)

			result[i].Contents = newContent
			break
		}
	}

	return result
}

// MergeConsecutiveToolMessages merges each run of consecutive tool messages into a single
// message, concatenating their contents in order. This is used for providers that count tool
// results per turn (Claude, Gemini): they require every result for one assistant turn to arrive
// in the next single turn, and reject a request where the results are spread over several turns.
//
//	Claude: "return one tool_result for each tool_use block, all together in the next user
//	message."
//	https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use
//
//	Gemini: rejects a mismatched turn with "Please ensure that the number of function response
//	parts is equal to the number of function call parts of the function call turn."
//	See the "Parallel function calling" section of
//	https://ai.google.dev/gemini-api/docs/function-calling and Google's sample, which passes
//	every result in one message:
//	https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/parallel_function_calling.ipynb
//
// Merging by run is safe because a new tool call cannot appear without an assistant message in
// between, so consecutive tool messages always answer the same call turn. A call that has no
// result still has none after merging, and the provider still rejects it.
//
// This must not be used for OpenAI, which requires one tool message per tool call ID.
//
// Name and Metadata are taken from the first message of the run. Neither the Claude nor the
// Gemini converter reads those fields.
func MergeConsecutiveToolMessages(messages []gollem.Message) []gollem.Message {
	hasRun := false
	for i := 1; i < len(messages); i++ {
		if messages[i].Role == gollem.RoleTool && messages[i-1].Role == gollem.RoleTool {
			hasRun = true
			break
		}
	}
	if !hasRun {
		return messages
	}

	result := make([]gollem.Message, 0, len(messages))
	for i := 0; i < len(messages); i++ {
		if messages[i].Role != gollem.RoleTool {
			result = append(result, messages[i])
			continue
		}

		// Copy the head of the run so the caller's messages are left untouched
		merged := messages[i]
		contents := make([]gollem.MessageContent, 0, len(merged.Contents))
		contents = append(contents, merged.Contents...)
		for i+1 < len(messages) && messages[i+1].Role == gollem.RoleTool {
			i++
			contents = append(contents, messages[i].Contents...)
		}
		merged.Contents = contents
		result = append(result, merged)
	}

	return result
}

// GenerateToolCallID generates a unique ID for tool calls if not present
func GenerateToolCallID(name string, index int) string {
	return "call_" + name + "_" + strconv.Itoa(index)
}
