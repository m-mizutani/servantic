package claude_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/m-mizutani/gt"
)

type complexTool struct{}

func (t *complexTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "complex_tool",
		Description: "A tool with complex parameter structure",
		Parameters: map[string]*gollem.Parameter{
			"user": {
				Type:     gollem.TypeObject,
				Required: true,
				Properties: map[string]*gollem.Parameter{
					"name": {
						Type:        gollem.TypeString,
						Description: "User's name",
						Required:    true,
					},
					"address": {
						Type: gollem.TypeObject,
						Properties: map[string]*gollem.Parameter{
							"street": {
								Type:        gollem.TypeString,
								Description: "Street address",
								Required:    true,
							},
							"city": {
								Type:        gollem.TypeString,
								Description: "City name",
							},
						},
					},
				},
			},
			"items": {
				Type: gollem.TypeArray,
				Items: &gollem.Parameter{
					Type: gollem.TypeObject,
					Properties: map[string]*gollem.Parameter{
						"id": {
							Type:        gollem.TypeString,
							Description: "Item ID",
						},
						"quantity": {
							Type:        gollem.TypeNumber,
							Description: "Item quantity",
						},
					},
				},
			},
		},
	}
}

func (t *complexTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return nil, nil
}

func TestConvertTool(t *testing.T) {
	tool := &complexTool{}
	claudeTool := claude.ConvertTool(tool)

	// Check basic properties
	gt.Equal(t, claudeTool.OfTool.Name, "complex_tool")

	// Check schema properties
	schemaProps := claudeTool.OfTool.InputSchema.Properties.(map[string]claude.JsonSchema)

	// Check user parameter
	user := schemaProps["user"]
	gt.Equal(t, user.Type, "object")

	userProps := user.Properties
	nameProps := userProps["name"]
	gt.Equal(t, nameProps.Type, "string")
	gt.Equal(t, nameProps.Description, "User's name")
	// Check that required array is generated from properties with Required=true
	gt.A(t, user.Required).Length(1)
	gt.Array(t, user.Required).Has("name")

	addressProps := userProps["address"].Properties
	gt.Equal(t, addressProps["street"].Type, "string")
	gt.Equal(t, addressProps["city"].Type, "string")

	// Check items parameter
	itemsProp := schemaProps["items"]
	gt.Equal(t, itemsProp.Type, "array")

	itemsSchema := *itemsProp.Items
	itemsProps := itemsSchema.Properties
	gt.Equal(t, itemsProps["id"].Type, "string")
	gt.Equal(t, itemsProps["quantity"].Type, "number")
}

func TestConvertParameterToSchema(t *testing.T) {
	type testCase struct {
		name     string
		schema   *gollem.Parameter
		expected claude.JsonSchema
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			actual := claude.ConvertParameterToSchema(tc.schema)
			gt.Value(t, actual).Equal(tc.expected)
		}
	}

	t.Run("number constraints", runTest(testCase{
		name: "number constraints",
		schema: &gollem.Parameter{
			Type:    gollem.TypeNumber,
			Minimum: ptr(1.0),
			Maximum: ptr(10.0),
		},
		expected: claude.JsonSchema{
			Type:    "number",
			Minimum: ptr(1.0),
			Maximum: ptr(10.0),
		},
	}))

	t.Run("string constraints", runTest(testCase{
		name: "string constraints",
		schema: &gollem.Parameter{
			Type:      gollem.TypeString,
			MinLength: ptr(1),
			MaxLength: ptr(10),
			Pattern:   "^[a-z]+$",
		},
		expected: claude.JsonSchema{
			Type:      "string",
			MinLength: ptr(1),
			MaxLength: ptr(10),
			Pattern:   "^[a-z]+$",
		},
	}))

	t.Run("array constraints", runTest(testCase{
		name: "array constraints",
		schema: &gollem.Parameter{
			Type:     gollem.TypeArray,
			Items:    &gollem.Parameter{Type: gollem.TypeString},
			MinItems: ptr(1),
			MaxItems: ptr(10),
		},
		expected: claude.JsonSchema{
			Type:     "array",
			MinItems: ptr(1),
			MaxItems: ptr(10),
			Items:    &claude.JsonSchema{Type: "string"},
		},
	}))

	t.Run("default value", runTest(testCase{
		name: "default value",
		schema: &gollem.Parameter{
			Type:    gollem.TypeString,
			Default: "default value",
		},
		expected: claude.JsonSchema{
			Type:    "string",
			Default: "default value",
		},
	}))
}

func ptr[T any](v T) *T {
	return &v
}

func TestConvertSchema(t *testing.T) {
	type testCase struct {
		name     string
		schema   *gollem.Parameter
		expected claude.JsonSchema
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			actual := claude.ConvertParameterToSchema(tc.schema)
			gt.Value(t, actual).Equal(tc.expected)
		}
	}

	t.Run("string type", runTest(testCase{
		name: "string type",
		schema: &gollem.Parameter{
			Type: gollem.TypeString,
		},
		expected: claude.JsonSchema{
			Type: "string",
		},
	}))
}

// multiRequiredTool has several required fields in one properties map so that a
// schema built from Go map iteration order would differ between conversions.
type multiRequiredTool struct{}

func (t *multiRequiredTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "multi_required_tool",
		Description: "A tool with several required parameters",
		Parameters: map[string]*gollem.Parameter{
			"zulu":  {Type: gollem.TypeString, Required: true},
			"alpha": {Type: gollem.TypeString, Required: true},
			"mike":  {Type: gollem.TypeString, Required: true},
			"bravo": {Type: gollem.TypeString, Required: true},
			"nested": {
				Type:     gollem.TypeObject,
				Required: true,
				Properties: map[string]*gollem.Parameter{
					"yankee": {Type: gollem.TypeString, Required: true},
					"delta":  {Type: gollem.TypeString, Required: true},
					"oscar":  {Type: gollem.TypeString, Required: true},
				},
			},
		},
	}
}

func (t *multiRequiredTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return nil, nil
}

// TestConvertToolIsByteStable pins the tool definition to be byte-identical
// between conversions of the same spec. Anthropic matches the prompt cache on an
// exact request prefix that begins with the tool definitions, so any field whose
// order comes from a Go map invalidates every cache breakpoint.
func TestConvertToolIsByteStable(t *testing.T) {
	tool := &multiRequiredTool{}

	first, err := json.Marshal(claude.ConvertTool(tool))
	gt.NoError(t, err)

	// Pin that a required array really is present in the marshalled bytes, so
	// the stability assertion below cannot pass vacuously. This is the nested
	// object's array: anthropic.ToolUnionParamOfTool only carries the
	// properties map, so the top-level required array built by
	// convertParametersToJSONSchema never reaches the request.
	gt.S(t, string(first)).Contains(`"required":["delta","oscar","yankee"]`)

	for i := 0; i < 100; i++ {
		actual, err := json.Marshal(claude.ConvertTool(tool))
		gt.NoError(t, err)
		gt.Equal(t, string(first), string(actual))
	}
}

func TestConvertParameterToSchemaRequiredIsSorted(t *testing.T) {
	schema := claude.ConvertParameterToSchema(&gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"zulu":  {Type: gollem.TypeString, Required: true},
			"alpha": {Type: gollem.TypeString, Required: true},
			"mike":  {Type: gollem.TypeString, Required: true},
		},
	})
	gt.Equal(t, []string{"alpha", "mike", "zulu"}, schema.Required)
}
