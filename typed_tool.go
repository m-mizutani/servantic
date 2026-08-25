package gollem

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/gollem-dev/gollem/internal/jsonutil"
	"github.com/m-mizutani/goerr/v2"
)

// NewTool creates a Tool from a typed handler. The input schema is inferred from
// the In type via ToSchema (so the same struct tags as Query[T] apply), and the
// handler receives a decoded In value instead of a raw map[string]any, removing
// the need for manual type assertions in Run.
//
// In must be a struct (its schema is a JSON "object" with declared properties).
// A map is rejected: it would yield a property-less schema that gives the LLM no
// argument information while providing none of the type safety this API exists for.
// Out must be a struct or map (it must encode to a JSON object); see toResultMap.
// If either type is unfit, an error is returned rather than a panic, so callers
// decide how to handle a malformed definition. Use MustNewTool for static
// registration where a panic is preferred.
func NewTool[In, Out any](
	name, description string,
	run func(ctx context.Context, in In) (Out, error),
) (Tool, error) {
	schema, err := buildToolSchema[In, Out]()
	if err != nil {
		return nil, goerr.Wrap(err, "failed to build tool", goerr.V("tool", name))
	}

	tool := &typedTool[In, Out]{
		name:        name,
		description: description,
		params:      schema.Properties,
		run:         run,
	}

	// Validate the resulting spec so that an empty name or otherwise invalid
	// definition is reported at construction time rather than at call time.
	spec := tool.Spec()
	if err := spec.Validate(); err != nil {
		return nil, goerr.Wrap(err, "invalid tool spec", goerr.V("tool", name))
	}

	return tool, nil
}

// MustNewTool is like NewTool but panics on error. It is intended for static
// initialization (e.g. package-level variables or slice literals passed to
// WithTools), where a malformed In/Out type is a programming error that should
// surface immediately. It follows the same convention as MustToSchema.
func MustNewTool[In, Out any](
	name, description string,
	run func(ctx context.Context, in In) (Out, error),
) Tool {
	tool, err := NewTool(name, description, run)
	if err != nil {
		panic(goerr.Wrap(err, "MustNewTool failed", goerr.V("tool", name)))
	}
	return tool
}

// ToolSchema builds the input parameter schema for a typed tool from the In type
// and validates that both In and Out can form a valid tool schema. The returned
// Parameter is the input object schema (TypeObject). Out is only validated — not
// returned — because ToolSpec has no output schema slot; the check ensures the
// handler's result can be converted to the wire map[string]any form.
func ToolSchema[In, Out any]() (*Parameter, error) {
	return buildToolSchema[In, Out]()
}

// MustToolSchema is like ToolSchema but panics on error, mirroring the
// ToSchema / MustToSchema pair. It is useful as a startup assertion of a typed
// tool's In/Out types.
func MustToolSchema[In, Out any]() *Parameter {
	schema, err := ToolSchema[In, Out]()
	if err != nil {
		panic(goerr.Wrap(err, "MustToolSchema failed"))
	}
	return schema
}

// typedTool adapts a typed handler func(ctx, In) (Out, error) to the Tool
// interface by inferring the schema from In and bridging args/result via JSON.
type typedTool[In, Out any] struct {
	name        string
	description string
	params      map[string]*Parameter
	run         func(ctx context.Context, in In) (Out, error)
}

func (t *typedTool[In, Out]) Spec() ToolSpec {
	return ToolSpec{
		Name:        t.name,
		Description: t.description,
		Parameters:  t.params,
	}
}

func (t *typedTool[In, Out]) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	// Bridge the untyped args into the typed In via a JSON round-trip. This uses
	// the same json tags that ToSchema reads for the schema, so field naming
	// cannot drift between the schema and the decode. Argument validation against
	// the schema already runs before Run (see executeToolCall); this decode is the
	// type-level safety net behind it.
	var in In
	raw, err := json.Marshal(args)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to encode tool arguments", goerr.V("tool", t.name))
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		return nil, goerr.Wrap(err, "failed to decode tool arguments", goerr.V("tool", t.name))
	}

	out, err := t.run(ctx, in)
	if err != nil {
		return nil, err
	}

	return toResultMap(out)
}

// buildToolSchema builds the input schema from In and verifies both In and Out.
// The In/Out shape is checked against the static type (reflect.TypeFor) rather
// than a zero value, so the verdict is deterministic even for types whose zero
// value is nil (e.g. interfaces or pointers) — a zero-value JSON probe would let
// those slip through. In must be a struct; Out must be a struct or map. The
// returned Parameter is the input object schema.
func buildToolSchema[In, Out any]() (*Parameter, error) {
	if err := assertObjectKind(reflect.TypeFor[In](), "In", false); err != nil {
		return nil, err
	}
	if err := assertObjectKind(reflect.TypeFor[Out](), "Out", true); err != nil {
		return nil, err
	}

	inSchema, err := ToSchema(*new(In))
	if err != nil {
		return nil, goerr.Wrap(err, "failed to build input schema", goerr.V("type", reflect.TypeFor[In]().String()))
	}

	return inSchema, nil
}

// assertObjectKind verifies that t (after unwrapping pointers) describes a JSON
// object. Structs always qualify; maps qualify only when allowMap is set (Out can
// be a map[string]any result, but a map In would produce a property-less schema).
func assertObjectKind(t reflect.Type, role string, allowMap bool) error {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.Struct:
		return nil
	case reflect.Map:
		if allowMap {
			return nil
		}
	}

	want := "a struct"
	if allowMap {
		want = "a struct or map"
	}
	return goerr.Wrap(ErrInvalidToolType, role+" must be "+want,
		goerr.V("type", t.String()), goerr.V("kind", t.Kind().String()))
}

// toResultMap converts a typed tool result into the wire map[string]any form. A
// map[string]any is returned as-is to preserve the existing tool style and avoid
// a needless round-trip; any other type is converted via JSON and therefore must
// encode to a JSON object (a struct or map), otherwise ErrInvalidToolType.
func toResultMap[Out any](out Out) (map[string]any, error) {
	if m, ok := any(out).(map[string]any); ok {
		return m, nil
	}

	raw, err := json.Marshal(out)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to encode tool result", goerr.V("type", fmt.Sprintf("%T", out)))
	}

	// Decode with json.Number so an int64 field of Out is not rounded to float64
	// on its way into the wire map.
	m, err := jsonutil.DecodeObject(raw)
	if err != nil {
		return nil, goerr.Wrap(ErrInvalidToolType, "tool result must encode to a JSON object",
			goerr.V("type", fmt.Sprintf("%T", out)))
	}
	return m, nil
}
