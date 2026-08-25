// Package gollem provides a unified interface for interacting with various LLM services.
package gollem

import (
	"context"
	"encoding/json"
	"reflect"

	"github.com/m-mizutani/goerr/v2"
)

// HistoryRepository is an interface for storing and loading conversation history.
// Implementations can use any storage backend (filesystem, S3, GCS, database, etc.).
type HistoryRepository interface {
	// Load retrieves a History by session ID.
	// Returns nil History and nil error if the session ID is not found.
	Load(ctx context.Context, sessionID string) (*History, error)

	// Save persists a History with the given session ID.
	// If a History already exists for the session ID, it is overwritten.
	Save(ctx context.Context, sessionID string, history *History) error
}

// History represents a conversation history that can be used across different LLM sessions.
// It stores messages in a format specific to each LLM type (OpenAI, Claude, or Gemini).
//
// For detailed documentation, see docs/history.md
type LLMType string

const (
	LLMTypeOpenAI LLMType = "OpenAI"
	LLMTypeGemini LLMType = "gemini"
	LLMTypeClaude LLMType = "claude"
)

const (
	HistoryVersion = 3 // Unified format version (v3: removed legacy function calls and provider dialects)
)

type History struct {
	LLType   LLMType   `json:"type"`
	Version  int       `json:"version"`
	Messages []Message `json:"messages"`
}

// UnmarshalJSON implements json.Unmarshaler with version validation.
// Returns ErrHistoryVersionMismatch if the serialized version does not match HistoryVersion.
func (x *History) UnmarshalJSON(data []byte) error {
	type historyAlias History
	var h historyAlias
	if err := json.Unmarshal(data, &h); err != nil {
		return err
	}

	if h.Version != HistoryVersion {
		return goerr.Wrap(ErrHistoryVersionMismatch, "unsupported history version",
			goerr.Value("got", h.Version),
			goerr.Value("want", HistoryVersion),
		)
	}

	*x = History(h)
	return nil
}

func (x *History) ToCount() int {
	if x == nil {
		return 0
	}
	return len(x.Messages)
}

func (x *History) Clone() *History {
	if x == nil {
		return nil
	}

	clone := &History{
		LLType:   x.LLType,
		Version:  x.Version,
		Messages: make([]Message, len(x.Messages)),
	}
	for i, msg := range x.Messages {
		clone.Messages[i] = cloneMessage(msg)
	}
	return clone
}

// cloneMessage returns a deep copy of m.
func cloneMessage(m Message) Message {
	clone := Message{
		Role: m.Role,
		Name: m.Name,
	}

	if m.Contents != nil {
		clone.Contents = make([]MessageContent, len(m.Contents))
		for i, c := range m.Contents {
			dataCopy := make(json.RawMessage, len(c.Data))
			copy(dataCopy, c.Data)
			var metaCopy json.RawMessage
			if c.Meta != nil {
				metaCopy = make(json.RawMessage, len(c.Meta))
				copy(metaCopy, c.Meta)
			}
			clone.Contents[i] = MessageContent{Type: c.Type, Data: dataCopy, Meta: metaCopy}
		}
	}

	if m.Metadata != nil {
		// Copy structurally rather than through a JSON round-trip. The round-trip could
		// fail (dropping Metadata silently, since Clone reports no error) and it rewrote
		// every number as a float64, so an int64 stored in Metadata came back rounded.
		clone.Metadata = cloneAnyMap(m.Metadata)
	}

	return clone
}

// cloneAnyMap returns a deep copy of m. Metadata is an exported map that callers fill with
// arbitrary Go values, not only values decoded from JSON, so the copy cannot assume the
// map[string]any / []any shapes: a []string or a map[string]string must be copied too, or
// the clone shares storage with the original.
func cloneAnyMap(m map[string]any) map[string]any {
	clone := make(map[string]any, len(m))
	for k, v := range m {
		clone[k] = cloneAnyValue(v)
	}
	return clone
}

func cloneAnyValue(v any) any {
	if v == nil {
		return nil
	}
	cloned := cloneReflect(reflect.ValueOf(v))
	if !cloned.IsValid() {
		return nil
	}
	return cloned.Interface()
}

// cloneReflect deep-copies the reference types a value can be built from. A pointer is
// followed so the clone does not alias the original's target. Channels and funcs have no
// meaningful copy and are carried over as-is, and so is a struct: its unexported fields
// cannot be set through reflection, and copying only the exported half would produce a
// value that is neither a real copy nor a plain assignment.
func cloneReflect(v reflect.Value) reflect.Value {
	switch v.Kind() {
	case reflect.Map:
		if v.IsNil() {
			return v
		}
		clone := reflect.MakeMapWithSize(v.Type(), v.Len())
		for _, key := range v.MapKeys() {
			clone.SetMapIndex(key, cloneReflect(v.MapIndex(key)))
		}
		return clone

	case reflect.Slice:
		if v.IsNil() {
			return v
		}
		clone := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		for i := 0; i < v.Len(); i++ {
			clone.Index(i).Set(cloneReflect(v.Index(i)))
		}
		return clone

	case reflect.Array:
		clone := reflect.New(v.Type()).Elem()
		for i := 0; i < v.Len(); i++ {
			clone.Index(i).Set(cloneReflect(v.Index(i)))
		}
		return clone

	case reflect.Pointer:
		if v.IsNil() {
			return v
		}
		clone := reflect.New(v.Type().Elem())
		clone.Elem().Set(cloneReflect(v.Elem()))
		return clone

	case reflect.Interface:
		if v.IsNil() {
			return v
		}
		clone := reflect.New(v.Type()).Elem()
		clone.Set(cloneReflect(v.Elem()))
		return clone

	default:
		return v
	}
}
