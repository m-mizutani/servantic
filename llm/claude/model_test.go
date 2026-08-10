package claude_test

import (
	"testing"

	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/m-mizutani/gt"
)

func TestNormalizeModelID(t *testing.T) {
	type testCase struct {
		input    string
		expected string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			gt.Equal(t, tc.expected, claude.NormalizeModelID(tc.input))
		}
	}

	t.Run("strips the dated suffix of the Claude API form", runTest(testCase{
		input:    "claude-sonnet-4-5-20250929",
		expected: "claude-sonnet-4-5",
	}))

	t.Run("strips the version separator of the Vertex AI form", runTest(testCase{
		input:    "claude-sonnet-4-5@20250929",
		expected: "claude-sonnet-4-5",
	}))

	t.Run("keeps the alias form as is", runTest(testCase{
		input:    "claude-sonnet-4-5",
		expected: "claude-sonnet-4-5",
	}))

	t.Run("keeps a dateless generation ID intact", runTest(testCase{
		input:    "claude-opus-4-8",
		expected: "claude-opus-4-8",
	}))

	t.Run("keeps a dateless single digit generation ID intact", runTest(testCase{
		input:    "claude-opus-5",
		expected: "claude-opus-5",
	}))

	t.Run("keeps a trailing segment that is not eight digits", runTest(testCase{
		input:    "claude-sonnet-4-5-2025092",
		expected: "claude-sonnet-4-5-2025092",
	}))

	t.Run("keeps a trailing segment that is not all digits", runTest(testCase{
		input:    "claude-sonnet-4-5-2025092a",
		expected: "claude-sonnet-4-5-2025092a",
	}))

	t.Run("keeps an ID with no separator", runTest(testCase{
		input:    "claude-2.1",
		expected: "claude-2.1",
	}))

	t.Run("keeps a non-date suffix after the version separator", runTest(testCase{
		input:    "claude-opus-5@custom",
		expected: "claude-opus-5@custom",
	}))

	t.Run("keeps an empty suffix after the version separator", runTest(testCase{
		input:    "claude-opus-5@",
		expected: "claude-opus-5@",
	}))

	t.Run("keeps an empty ID", runTest(testCase{
		input:    "",
		expected: "",
	}))
}

func TestResolveMaxOutputTokens(t *testing.T) {
	type testCase struct {
		model    string
		expected int64
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			gt.Equal(t, tc.expected, claude.ResolveMaxOutputTokens(tc.model))
		}
	}

	// Every model documented at
	// https://platform.claude.com/docs/en/about-claude/models/overview
	t.Run("claude-fable-5", runTest(testCase{model: "claude-fable-5", expected: 128000}))
	t.Run("claude-mythos-5", runTest(testCase{model: "claude-mythos-5", expected: 128000}))
	t.Run("claude-opus-5", runTest(testCase{model: "claude-opus-5", expected: 128000}))
	t.Run("claude-opus-4-8", runTest(testCase{model: "claude-opus-4-8", expected: 128000}))
	t.Run("claude-opus-4-7", runTest(testCase{model: "claude-opus-4-7", expected: 128000}))
	t.Run("claude-opus-4-6", runTest(testCase{model: "claude-opus-4-6", expected: 128000}))
	t.Run("claude-opus-4-5", runTest(testCase{model: "claude-opus-4-5", expected: 64000}))
	t.Run("claude-sonnet-5", runTest(testCase{model: "claude-sonnet-5", expected: 128000}))
	t.Run("claude-sonnet-4-6", runTest(testCase{model: "claude-sonnet-4-6", expected: 128000}))
	t.Run("claude-sonnet-4-5", runTest(testCase{model: "claude-sonnet-4-5", expected: 64000}))
	t.Run("claude-haiku-4-5", runTest(testCase{model: "claude-haiku-4-5", expected: 64000}))

	// The dated Claude API form resolves to the same limit as its alias.
	t.Run("dated opus 4.5", runTest(testCase{model: "claude-opus-4-5-20251101", expected: 64000}))
	t.Run("dated sonnet 4.5", runTest(testCase{model: "claude-sonnet-4-5-20250929", expected: 64000}))
	t.Run("dated haiku 4.5", runTest(testCase{model: "claude-haiku-4-5-20251001", expected: 64000}))

	// The Vertex AI form resolves to the same limit as its alias.
	t.Run("vertex opus 4.5", runTest(testCase{model: "claude-opus-4-5@20251101", expected: 64000}))
	t.Run("vertex sonnet 4.5", runTest(testCase{model: "claude-sonnet-4-5@20250929", expected: 64000}))
	t.Run("vertex opus 5", runTest(testCase{model: "claude-opus-5@20260724", expected: 128000}))

	// A suffix that is not an 8 digit date keeps the ID distinct from the base
	// model, so it falls back instead of inheriting the base model's limit.
	t.Run("non-date version suffix", runTest(testCase{
		model:    "claude-opus-5@custom",
		expected: claude.FallbackMaxOutputTokens,
	}))

	// Models absent from the table fall back to the smallest documented limit.
	t.Run("unknown model", runTest(testCase{
		model:    "my-proxy/llm",
		expected: claude.FallbackMaxOutputTokens,
	}))
	t.Run("empty model", runTest(testCase{
		model:    "",
		expected: claude.FallbackMaxOutputTokens,
	}))
	t.Run("retired model kept out of the table", runTest(testCase{
		model:    "claude-sonnet-4@20250514",
		expected: claude.FallbackMaxOutputTokens,
	}))

	// Any 8 digit suffix is treated as a snapshot of the base model. The date
	// is not validated: an ID the API rejects fails at the API, and one that
	// resolves too high fails with the API's own max_tokens error.
	t.Run("unvalidated date suffix resolves to the base model", runTest(testCase{
		model:    "claude-opus-5-99999999",
		expected: 128000,
	}))

	t.Run("fallback is the smallest documented limit", func(t *testing.T) {
		gt.Equal(t, int64(64000), claude.FallbackMaxOutputTokens)
	})
}
