package claude

import (
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gollem-dev/gollem"
	gollemschema "github.com/gollem-dev/gollem/internal/schema"
)

func convertTool(tool gollem.Tool) anthropic.ToolUnionParam {
	spec := tool.Spec()
	schema := convertParametersToJSONSchema(spec.Parameters)

	// ToolUnionParamOfTool only fills InputSchema and Name, so the description
	// and the top-level required array have to be set on the variant afterwards.
	// Without them the model receives the properties alone and is told neither
	// what the tool does nor which arguments are mandatory.
	toolParam := anthropic.ToolUnionParamOfTool(
		anthropic.ToolInputSchemaParam{
			Properties: schema.Properties,
			Required:   schema.Required,
		},
		spec.Name,
	)
	if spec.Description != "" {
		toolParam.OfTool.Description = anthropic.String(spec.Description)
	}

	return toolParam
}

type jsonSchema struct {
	Type        string                `json:"type"`
	Properties  map[string]jsonSchema `json:"properties,omitempty"`
	Required    []string              `json:"required,omitempty"`
	Items       *jsonSchema           `json:"items,omitempty"`
	Minimum     *float64              `json:"minimum,omitempty"`
	Maximum     *float64              `json:"maximum,omitempty"`
	MinLength   *int                  `json:"minLength,omitempty"`
	MaxLength   *int                  `json:"maxLength,omitempty"`
	Pattern     string                `json:"pattern,omitempty"`
	MinItems    *int                  `json:"minItems,omitempty"`
	MaxItems    *int                  `json:"maxItems,omitempty"`
	Default     interface{}           `json:"default,omitempty"`
	Enum        []interface{}         `json:"enum,omitempty"`
	Description string                `json:"description,omitempty"`
	Title       string                `json:"title,omitempty"`
}

func convertParametersToJSONSchema(params map[string]*gollem.Parameter) jsonSchema {
	properties := make(map[string]jsonSchema)

	for name, param := range params {
		properties[name] = convertParameterToSchema(param)
	}

	schema := jsonSchema{
		Type:       "object",
		Properties: properties,
	}

	if required := gollemschema.CollectRequiredFields(params); len(required) > 0 {
		schema.Required = required
	}

	return schema
}

// convertParameterToSchema converts gollem.Parameter to Claude schema
func convertParameterToSchema(param *gollem.Parameter) jsonSchema {
	schema := jsonSchema{
		Type:        getClaudeType(param.Type),
		Description: param.Description,
		Title:       param.Title,
	}

	if len(param.Enum) > 0 {
		enum := make([]interface{}, len(param.Enum))
		for i, v := range param.Enum {
			enum[i] = v
		}
		schema.Enum = enum
	}

	if param.Properties != nil {
		properties := make(map[string]jsonSchema)
		for name, prop := range param.Properties {
			properties[name] = convertParameterToSchema(prop)
		}
		schema.Properties = properties
		if required := gollemschema.CollectRequiredFields(param.Properties); len(required) > 0 {
			schema.Required = required
		}
	}

	if param.Items != nil {
		items := convertParameterToSchema(param.Items)
		schema.Items = &items
	}

	// Add number constraints
	if param.Type == gollem.TypeNumber || param.Type == gollem.TypeInteger {
		if param.Minimum != nil {
			schema.Minimum = param.Minimum
		}
		if param.Maximum != nil {
			schema.Maximum = param.Maximum
		}
	}

	// Add string constraints
	if param.Type == gollem.TypeString {
		if param.MinLength != nil {
			schema.MinLength = param.MinLength
		}
		if param.MaxLength != nil {
			schema.MaxLength = param.MaxLength
		}
		if param.Pattern != "" {
			schema.Pattern = param.Pattern
		}
	}

	// Add array constraints
	if param.Type == gollem.TypeArray {
		if param.MinItems != nil {
			schema.MinItems = param.MinItems
		}
		if param.MaxItems != nil {
			schema.MaxItems = param.MaxItems
		}
	}

	// Add default value
	if param.Default != nil {
		schema.Default = param.Default
	}

	return schema
}

func getClaudeType(paramType gollem.ParameterType) string {
	switch paramType {
	case gollem.TypeString:
		return "string"
	case gollem.TypeNumber:
		return "number"
	case gollem.TypeInteger:
		return "integer"
	case gollem.TypeBoolean:
		return "boolean"
	case gollem.TypeArray:
		return "array"
	case gollem.TypeObject:
		return "object"
	default:
		return "string"
	}
}
