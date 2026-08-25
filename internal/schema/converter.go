package schema

import (
	"encoding/json"
	"slices"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

// CollectRequiredFields returns a list of required property names in ascending
// name order. The order must not depend on Go map iteration: the result is
// emitted verbatim as the JSON Schema "required" array, and Anthropic matches
// the prompt cache on an exact byte prefix that starts with the tool
// definitions, so a reordered array invalidates every cache breakpoint.
func CollectRequiredFields(properties map[string]*gollem.Parameter) []string {
	var required []string
	for name, prop := range properties {
		if prop.Required {
			required = append(required, name)
		}
	}
	slices.Sort(required)
	return required
}

// ConvertParameterToJSONSchema converts gollem.Parameter to JSON Schema map
// This is the base conversion without provider-specific modifications
func ConvertParameterToJSONSchema(param *gollem.Parameter) map[string]any {
	schema := map[string]any{
		"type": string(param.Type),
	}

	if param.Description != "" {
		schema["description"] = param.Description
	}

	if param.Type == gollem.TypeObject && param.Properties != nil {
		props := make(map[string]any)
		for name, prop := range param.Properties {
			props[name] = ConvertParameterToJSONSchema(prop)
		}
		schema["properties"] = props
		schema["additionalProperties"] = false

		// Collect required fields from properties
		if required := CollectRequiredFields(param.Properties); len(required) > 0 {
			schema["required"] = required
		}
	}

	if param.Type == gollem.TypeArray && param.Items != nil {
		schema["items"] = ConvertParameterToJSONSchema(param.Items)
	}

	if param.Enum != nil {
		schema["enum"] = param.Enum
	}

	// Add constraints
	if param.Minimum != nil {
		schema["minimum"] = *param.Minimum
	}
	if param.Maximum != nil {
		schema["maximum"] = *param.Maximum
	}
	if param.MinLength != nil {
		schema["minLength"] = *param.MinLength
	}
	if param.MaxLength != nil {
		schema["maxLength"] = *param.MaxLength
	}
	if param.Pattern != "" {
		schema["pattern"] = param.Pattern
	}
	if param.MinItems != nil {
		schema["minItems"] = *param.MinItems
	}
	if param.MaxItems != nil {
		schema["maxItems"] = *param.MaxItems
	}

	return schema
}

// ConvertParameterToJSONString converts Parameter to a JSON Schema string
// This is used by Claude for embedding schema in system prompt
func ConvertParameterToJSONString(param *gollem.Parameter) (string, error) {
	if param == nil {
		return "", nil
	}

	// Validate schema
	if err := param.Validate(); err != nil {
		return "", goerr.Wrap(err, "invalid response schema")
	}

	// Build JSON Schema object
	schemaObj := map[string]any{
		"type":    "object",
		"$schema": "http://json-schema.org/draft-07/schema#",
	}

	if param.Description != "" {
		schemaObj["description"] = param.Description
	}

	// Convert Parameter to JSON Schema
	innerSchema := ConvertParameterToJSONSchema(param)

	// Merge properties from inner schema
	for k, v := range innerSchema {
		schemaObj[k] = v
	}

	// Marshal to pretty JSON
	schemaJSON, err := json.MarshalIndent(schemaObj, "", "  ")
	if err != nil {
		return "", goerr.Wrap(err, "failed to marshal schema")
	}

	return string(schemaJSON), nil
}
