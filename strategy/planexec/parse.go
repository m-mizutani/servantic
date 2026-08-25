package planexec

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/gollem-dev/gollem"
)

// formatToolResult formats tool execution result as a JSON string
func formatToolResult(result map[string]any) string {
	// Empty result
	if len(result) == 0 {
		return ""
	}

	// Try to format as indented JSON
	jsonBytes, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		// Fallback: use Go's default formatting
		return fmt.Sprintf("Tool Result: %v", result)
	}

	return fmt.Sprintf("Tool Result:\n%s", string(jsonBytes))
}

// parseTaskResult extracts the task execution result from LLM response and tool results
func parseTaskResult(response *gollem.Response, nextInput []gollem.Input) string {
	if response == nil {
		return ""
	}

	// Combine all text responses
	var results []string
	for _, text := range response.Texts {
		text = strings.TrimSpace(text)
		if text != "" {
			results = append(results, text)
		}
	}

	// If there are function calls, include them as well
	if len(response.FunctionCalls) > 0 {
		results = append(results, "Tool calls executed:")
		for _, fc := range response.FunctionCalls {
			results = append(results, "- "+fc.Name)
			// Include arguments if any. The names are sorted because this text becomes
			// Task.Result and is embedded verbatim into the plan and reflect prompts;
			// map iteration order would make the same tool call render differently on
			// every run.
			if len(fc.Arguments) > 0 {
				names := make([]string, 0, len(fc.Arguments))
				for key := range fc.Arguments {
					names = append(names, key)
				}
				slices.Sort(names)
				for _, key := range names {
					results = append(results, fmt.Sprintf("  %s: %v", key, fc.Arguments[key]))
				}
			}
		}
	}

	// Extract tool execution results from nextInput
	for _, input := range nextInput {
		if funcResp, ok := input.(gollem.FunctionResponse); ok {
			if funcResp.Data != nil {
				formatted := formatToolResult(funcResp.Data)
				if formatted != "" {
					results = append(results, formatted)
				}
			}
		}
	}

	return strings.Join(results, "\n\n")
}
