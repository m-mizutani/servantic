package claude

import "strings"

// maxOutputTokens maps a normalized Claude model ID to the maximum value the
// Messages API accepts for max_tokens on that model.
//
// Source: https://platform.claude.com/docs/en/about-claude/models/overview
// (retrieved 2026-08-10). Retired models are intentionally absent: requests to
// them fail regardless of max_tokens, so carrying a limit for them is pointless.
var maxOutputTokens = map[string]int64{
	"claude-fable-5":    128000,
	"claude-mythos-5":   128000,
	"claude-opus-5":     128000,
	"claude-opus-4-8":   128000,
	"claude-opus-4-7":   128000,
	"claude-opus-4-6":   128000,
	"claude-opus-4-5":   64000,
	"claude-sonnet-5":   128000,
	"claude-sonnet-4-6": 128000,
	"claude-sonnet-4-5": 64000,
	"claude-haiku-4-5":  64000,
}

// fallbackMaxOutputTokens is used for models absent from maxOutputTokens, such
// as a model released after this table was written or a model served through a
// compatible endpoint configured with WithBaseURL.
//
// It is the smallest limit in the table, so every currently documented model
// accepts it. A model whose real limit is lower will have the request rejected
// by the API; callers in that situation must pass WithMaxTokens explicitly.
const fallbackMaxOutputTokens int64 = 64000

// normalizeModelID reduces the Claude API dated form, the alias form and the
// Vertex AI form of a model ID to a single key. Only an 8 digit date suffix is
// stripped, so a suffix that carries some other meaning keeps the ID distinct
// from the base model rather than silently inheriting its limit.
//
//	claude-sonnet-4-5-20250929 -> claude-sonnet-4-5
//	claude-sonnet-4-5@20250929 -> claude-sonnet-4-5
//	claude-opus-4-8            -> claude-opus-4-8
//	claude-opus-5@custom       -> claude-opus-5@custom
func normalizeModelID(model string) string {
	if i := strings.IndexByte(model, '@'); i >= 0 && isDateSuffix(model[i+1:]) {
		model = model[:i]
	}
	if i := strings.LastIndexByte(model, '-'); i >= 0 {
		if isDateSuffix(model[i+1:]) {
			model = model[:i]
		}
	}
	return model
}

// isDateSuffix reports whether s is an 8 digit date such as "20250929". Model
// IDs from the 4.6 generation onward are dateless (claude-opus-4-8), so the
// length check is what keeps their trailing segment intact.
func isDateSuffix(s string) bool {
	if len(s) != 8 {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// resolveMaxOutputTokens returns the maximum output tokens documented for the
// given model, or fallbackMaxOutputTokens when the model is not in the table.
//
// The ID is looked up verbatim before it is normalized, so a future table entry
// whose own ID ends in an 8 digit segment stays reachable.
func resolveMaxOutputTokens(model string) int64 {
	if v, ok := maxOutputTokens[model]; ok {
		return v
	}
	if v, ok := maxOutputTokens[normalizeModelID(model)]; ok {
		return v
	}
	return fallbackMaxOutputTokens
}
