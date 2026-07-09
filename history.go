// Package gollem provides a unified interface for interacting with various LLM services.
package gollem

import (
	"context"
	"encoding/json"

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
		// Use JSON round-trip to deep-copy Metadata values, which may themselves be
		// reference types (maps or slices).
		if data, err := json.Marshal(m.Metadata); err == nil {
			var metaCopy map[string]interface{}
			if err := json.Unmarshal(data, &metaCopy); err == nil {
				clone.Metadata = metaCopy
			}
		}
		// If marshal/unmarshal fails (should not happen for well-formed metadata),
		// clone.Metadata remains nil rather than sharing the original's references.
	}

	return clone
}
