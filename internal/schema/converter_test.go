package schema_test

import (
	"encoding/json"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/internal/schema"
	"github.com/m-mizutani/gt"
)

// newMultiRequiredParameter returns an object parameter whose properties contain
// several required fields at both the top level and a nested level. Map
// iteration order is randomized per range statement, so a schema built from it
// is a reliable way to observe non-deterministic ordering.
func newMultiRequiredParameter() *gollem.Parameter {
	return &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"zulu":    {Type: gollem.TypeString, Required: true},
			"alpha":   {Type: gollem.TypeString, Required: true},
			"mike":    {Type: gollem.TypeString, Required: true},
			"bravo":   {Type: gollem.TypeString, Required: true},
			"charlie": {Type: gollem.TypeString},
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

func TestCollectRequiredFields(t *testing.T) {
	t.Run("returns required names in ascending order", func(t *testing.T) {
		param := newMultiRequiredParameter()
		gt.Equal(t, []string{"alpha", "bravo", "mike", "nested", "zulu"},
			schema.CollectRequiredFields(param.Properties))
	})

	t.Run("returns the same order on every call", func(t *testing.T) {
		param := newMultiRequiredParameter()
		first := schema.CollectRequiredFields(param.Properties)
		for i := 0; i < 100; i++ {
			gt.Equal(t, first, schema.CollectRequiredFields(param.Properties))
		}
	})

	t.Run("returns nil when no property is required", func(t *testing.T) {
		properties := map[string]*gollem.Parameter{
			"alpha": {Type: gollem.TypeString},
		}
		gt.Nil(t, schema.CollectRequiredFields(properties))
	})
}

func TestConvertParameterToJSONSchemaIsByteStable(t *testing.T) {
	param := newMultiRequiredParameter()

	first, err := json.Marshal(schema.ConvertParameterToJSONSchema(param))
	gt.NoError(t, err)

	for i := 0; i < 100; i++ {
		actual, err := json.Marshal(schema.ConvertParameterToJSONSchema(param))
		gt.NoError(t, err)
		gt.Equal(t, string(first), string(actual))
	}
}

func TestConvertParameterToJSONStringIsByteStable(t *testing.T) {
	param := newMultiRequiredParameter()

	first, err := schema.ConvertParameterToJSONString(param)
	gt.NoError(t, err)

	for i := 0; i < 100; i++ {
		actual, err := schema.ConvertParameterToJSONString(param)
		gt.NoError(t, err)
		gt.Equal(t, first, actual)
	}
}
