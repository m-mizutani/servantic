package gollem

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/m-mizutani/goerr/v2"
)

// ToolSpec is the specification of a tool.
type ToolSpec struct {
	Name        string
	Description string
	Parameters  map[string]*Parameter
}

// ValidateArgs validates the given arguments against the tool's parameter specifications.
// It checks all parameters and collects all validation errors.
// Returns nil if all arguments are valid.
func (s *ToolSpec) ValidateArgs(args map[string]any) error {
	// Sort parameter names for deterministic error ordering
	names := make([]string, 0, len(s.Parameters))
	for name := range s.Parameters {
		names = append(names, name)
	}
	slices.Sort(names)

	var errs []error
	for _, name := range names {
		param := s.Parameters[name]
		if err := param.ValidateValue(name, args[name]); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return &toolArgsValidationError{
			toolName: s.Name,
			errs:     errs,
		}
	}
	return nil
}

// toolArgsValidationError holds multiple argument validation errors for a tool.
// It formats a structured error message that helps LLMs understand and correct invalid arguments.
type toolArgsValidationError struct {
	toolName string
	errs     []error
}

func (e *toolArgsValidationError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Tool argument validation failed for %q:\n", e.toolName)
	for _, err := range e.errs {
		fmt.Fprintf(&b, "  - %s\n", err.Error())
	}
	b.WriteString("Please correct the arguments and retry the tool call.")
	return b.String()
}

func (e *toolArgsValidationError) Unwrap() error {
	return ErrToolArgsValidation
}

// Validate validates the tool specification.
func (s *ToolSpec) Validate() error {
	eb := goerr.NewBuilder(goerr.V("tool", s))
	if s.Name == "" {
		return eb.Wrap(ErrInvalidTool, "name is required")
	}

	paramNames := make(map[string]struct{})
	for name, param := range s.Parameters {
		if _, ok := paramNames[name]; ok {
			return eb.Wrap(ErrInvalidTool, "duplicate parameter name", goerr.V("name", name))
		}
		paramNames[name] = struct{}{}

		if err := param.Validate(); err != nil {
			return eb.Wrap(ErrInvalidTool, "invalid parameter")
		}
	}

	return nil
}

// ParameterType is the type of a parameter.
type ParameterType string

const (
	TypeString  ParameterType = "string"
	TypeNumber  ParameterType = "number"
	TypeInteger ParameterType = "integer"
	TypeBoolean ParameterType = "boolean"
	TypeArray   ParameterType = "array"
	TypeObject  ParameterType = "object"
)

// Parameter is a parameter of a tool.
type Parameter struct {
	// Title is the user friendly  of the parameter. It's optional.
	Title string

	// Type is the type of the parameter. It's required.
	Type ParameterType

	// Description is the description of the parameter. It's optional.
	Description string

	// Required indicates if this parameter is required.
	Required bool

	// Enum is the enum of the parameter. It's optional.
	Enum []string

	// Properties is the properties of the parameter. It's used for object type.
	Properties map[string]*Parameter

	// Items is the items of the parameter. It's used for array type.
	Items *Parameter

	// Number constraints
	Minimum *float64
	Maximum *float64

	// String constraints
	MinLength *int
	MaxLength *int
	Pattern   string

	// Array constraints
	MinItems *int
	MaxItems *int

	// Default value
	Default any
}

// Validate validates the parameter.
func (p *Parameter) Validate() error {
	eb := goerr.NewBuilder(goerr.V("parameter", p))

	// Type is required
	if p.Type == "" {
		return eb.Wrap(ErrInvalidParameter, "type is required")
	}

	// Validate parameter type
	switch p.Type {
	case TypeString, TypeNumber, TypeInteger, TypeBoolean, TypeArray, TypeObject:
		// Valid type
	default:
		return eb.Wrap(ErrInvalidParameter, "invalid parameter type", goerr.V("type", p.Type))
	}

	// Properties is required for object type
	if p.Type == TypeObject {
		if p.Properties == nil {
			return eb.Wrap(ErrInvalidParameter, "properties is required for object type")
		}

		// Check for duplicate property names
		propNames := make(map[string]struct{})
		for name := range p.Properties {
			if _, ok := propNames[name]; ok {
				return eb.Wrap(ErrInvalidParameter, "duplicate property name", goerr.V("name", name))
			}
			propNames[name] = struct{}{}
		}

		// Validate nested properties
		for _, prop := range p.Properties {
			if err := prop.Validate(); err != nil {
				return eb.Wrap(ErrInvalidParameter, "invalid property")
			}
		}
	}

	// Items is required for array type
	if p.Type == TypeArray {
		if p.Items == nil {
			return eb.Wrap(ErrInvalidParameter, "items is required for array type")
		}
		// Validate items
		if err := p.Items.Validate(); err != nil {
			return eb.Wrap(ErrInvalidParameter, "invalid items")
		}
	}

	// Validate number constraints
	if p.Type == TypeNumber || p.Type == TypeInteger {
		if p.Minimum != nil && p.Maximum != nil && *p.Minimum > *p.Maximum {
			return eb.Wrap(ErrInvalidParameter, "minimum must be less than or equal to maximum")
		}
	}

	// Validate string constraints
	if p.Type == TypeString {
		if p.MinLength != nil && p.MaxLength != nil && *p.MinLength > *p.MaxLength {
			return eb.Wrap(ErrInvalidParameter, "minLength must be less than or equal to maxLength")
		}
		if p.Pattern != "" {
			if _, err := regexp.Compile(p.Pattern); err != nil {
				return eb.Wrap(ErrInvalidParameter, "invalid pattern", goerr.V("pattern", p.Pattern))
			}
		}
	}

	// Validate array constraints
	if p.Type == TypeArray {
		if p.MinItems != nil && p.MaxItems != nil && *p.MinItems > *p.MaxItems {
			return eb.Wrap(ErrInvalidParameter, "minItems must be less than or equal to maxItems")
		}
	}

	return nil
}

// ValidateValue validates a value against this parameter's specification.
// It checks required, type, enum, and constraint validations.
// Returns nil if the value is valid, or an error describing the validation failure.
func (p *Parameter) ValidateValue(name string, value any) error {
	eb := goerr.NewBuilder(goerr.V("parameter", name))

	// Check required
	if value == nil {
		if p.Required {
			return eb.Wrap(ErrInvalidParameter, "required parameter missing")
		}
		return nil // Optional parameter with no value is valid
	}

	// Type validation
	switch p.Type {
	case TypeString:
		s, ok := value.(string)
		if !ok {
			return eb.Wrap(ErrInvalidParameter, "expected string type", goerr.V("actual", value))
		}
		// Enum validation
		if len(p.Enum) > 0 && !slices.Contains(p.Enum, s) {
			return eb.Wrap(ErrInvalidParameter, "value not in enum", goerr.V("value", s), goerr.V("enum", p.Enum))
		}
		// String length validation
		if p.MinLength != nil && len(s) < *p.MinLength {
			return eb.Wrap(ErrInvalidParameter, "string too short", goerr.V("minLength", *p.MinLength), goerr.V("actual", len(s)))
		}
		if p.MaxLength != nil && len(s) > *p.MaxLength {
			return eb.Wrap(ErrInvalidParameter, "string too long", goerr.V("maxLength", *p.MaxLength), goerr.V("actual", len(s)))
		}
		// Pattern validation
		if p.Pattern != "" {
			matched, err := regexp.MatchString(p.Pattern, s)
			if err != nil {
				return eb.Wrap(err, "pattern matching failed")
			}
			if !matched {
				return eb.Wrap(ErrInvalidParameter, "string does not match pattern", goerr.V("pattern", p.Pattern))
			}
		}

	case TypeNumber:
		var n float64
		switch v := value.(type) {
		case float64:
			n = v
		case float32:
			n = float64(v)
		case int:
			n = float64(v)
		case int64:
			n = float64(v)
		case json.Number:
			// An argument too wide for float64 is decoded as json.Number to keep its exact
			// value; the range check below is still done in float64, which is enough to
			// compare against Minimum and Maximum.
			f, err := v.Float64()
			if err != nil {
				return eb.Wrap(ErrInvalidParameter, "expected number type", goerr.V("actual", value))
			}
			n = f
		default:
			return eb.Wrap(ErrInvalidParameter, "expected number type", goerr.V("actual", value))
		}
		if p.Minimum != nil && n < *p.Minimum {
			return eb.Wrap(ErrInvalidParameter, "number too small", goerr.V("minimum", *p.Minimum), goerr.V("actual", n))
		}
		if p.Maximum != nil && n > *p.Maximum {
			return eb.Wrap(ErrInvalidParameter, "number too large", goerr.V("maximum", *p.Maximum), goerr.V("actual", n))
		}

	case TypeInteger:
		var n int64
		switch v := value.(type) {
		case int:
			n = int64(v)
		case int64:
			n = v
		case float64:
			if v != float64(int64(v)) {
				return eb.Wrap(ErrInvalidParameter, "expected integer type, got float", goerr.V("actual", v))
			}
			n = int64(v)
		case json.Number:
			// An integer too wide for float64 is decoded as json.Number to keep its exact
			// value, so it is parsed as an int64 rather than through float64.
			i, err := v.Int64()
			if err != nil {
				return eb.Wrap(ErrInvalidParameter, "expected integer type", goerr.V("actual", value))
			}
			n = i
		default:
			return eb.Wrap(ErrInvalidParameter, "expected integer type", goerr.V("actual", value))
		}
		if p.Minimum != nil && float64(n) < *p.Minimum {
			return eb.Wrap(ErrInvalidParameter, "integer too small", goerr.V("minimum", *p.Minimum), goerr.V("actual", n))
		}
		if p.Maximum != nil && float64(n) > *p.Maximum {
			return eb.Wrap(ErrInvalidParameter, "integer too large", goerr.V("maximum", *p.Maximum), goerr.V("actual", n))
		}

	case TypeBoolean:
		if _, ok := value.(bool); !ok {
			return eb.Wrap(ErrInvalidParameter, "expected boolean type", goerr.V("actual", value))
		}

	case TypeArray:
		arr, ok := value.([]any)
		if !ok {
			return eb.Wrap(ErrInvalidParameter, "expected array type", goerr.V("actual", value))
		}
		if p.MinItems != nil && len(arr) < *p.MinItems {
			return eb.Wrap(ErrInvalidParameter, "array too short", goerr.V("minItems", *p.MinItems), goerr.V("actual", len(arr)))
		}
		if p.MaxItems != nil && len(arr) > *p.MaxItems {
			return eb.Wrap(ErrInvalidParameter, "array too long", goerr.V("maxItems", *p.MaxItems), goerr.V("actual", len(arr)))
		}
		// Validate each item if Items schema is defined
		if p.Items != nil {
			for i, item := range arr {
				if err := p.Items.ValidateValue(name+"["+strconv.Itoa(i)+"]", item); err != nil {
					return err
				}
			}
		}

	case TypeObject:
		obj, ok := value.(map[string]any)
		if !ok {
			return eb.Wrap(ErrInvalidParameter, "expected object type", goerr.V("actual", value))
		}
		// Validate each property if Properties schema is defined
		if p.Properties != nil {
			for propName, propParam := range p.Properties {
				propValue := obj[propName]
				if err := propParam.ValidateValue(name+"."+propName, propValue); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// Tool is specification and execution of an action that can be called by the LLM.
type Tool interface {
	// Spec returns the specification of the tool. It's called when starting a LLM chat session in Prompt().
	Spec() ToolSpec

	// Run is the execution of the tool.
	// It's called when receiving a tool call from the LLM. Even if the method returns an error, the tool execution is not aborted. Error will be passed to LLM as a response. If you want to abort the tool execution, you need to return an error from the callback function of WithToolErrorHook().
	// Special case: If the tool returns ErrExitConversation, the conversation loop will be terminated normally and the session will be completed successfully.
	Run(ctx context.Context, args map[string]any) (map[string]any, error)
}

// ToolSet is a set of tools.
// It's useful for providing a set of tools to the LLM.
type ToolSet interface {
	// Specs returns the specifications of the tools.
	Specs(ctx context.Context) ([]ToolSpec, error)

	// Run is the execution of the tool.
	// It's called when receiving a tool call from the LLM.
	Run(ctx context.Context, name string, args map[string]any) (map[string]any, error)
}
