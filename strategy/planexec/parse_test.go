package planexec_test

import (
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/strategy/planexec"
	"github.com/m-mizutani/gt"
)

func TestFormatToolResult(t *testing.T) {
	runTest := func(tc struct {
		name     string
		input    map[string]any
		expected string
	}) func(t *testing.T) {
		return func(t *testing.T) {
			result := planexec.FormatToolResult(tc.input)
			if tc.expected == "" {
				gt.V(t, result).Equal("")
			} else {
				gt.S(t, result).Contains(tc.expected)
			}
		}
	}

	t.Run("empty result", runTest(struct {
		name     string
		input    map[string]any
		expected string
	}{
		name:     "empty result",
		input:    map[string]any{},
		expected: "",
	}))

	t.Run("basic result", runTest(struct {
		name     string
		input    map[string]any
		expected string
	}{
		name: "basic result",
		input: map[string]any{
			"status": "ok",
			"count":  42,
		},
		expected: "status",
	}))

	t.Run("with records array", runTest(struct {
		name     string
		input    map[string]any
		expected string
	}{
		name: "with records array",
		input: map[string]any{
			"records": []map[string]any{
				{"id": 1, "name": "Alice"},
				{"id": 2, "name": "Bob"},
			},
			"count": 2,
		},
		expected: "records",
	}))
}

func TestParseTaskResult(t *testing.T) {
	runTest := func(tc struct {
		name      string
		response  *gollem.Response
		nextInput []gollem.Input
		expected  []string
	}) func(t *testing.T) {
		return func(t *testing.T) {
			result := planexec.ParseTaskResult(tc.response, tc.nextInput)
			for _, exp := range tc.expected {
				gt.S(t, result).Contains(exp)
			}
		}
	}

	t.Run("only texts", runTest(struct {
		name      string
		response  *gollem.Response
		nextInput []gollem.Input
		expected  []string
	}{
		name: "only texts",
		response: &gollem.Response{
			Texts: []string{"Result: Success"},
		},
		nextInput: []gollem.Input{},
		expected:  []string{"Result: Success"},
	}))

	t.Run("with tool results", runTest(struct {
		name      string
		response  *gollem.Response
		nextInput []gollem.Input
		expected  []string
	}{
		name: "with tool results",
		response: &gollem.Response{
			Texts: []string{"Query executed"},
		},
		nextInput: []gollem.Input{
			gollem.FunctionResponse{
				ID:   "call_1",
				Name: "query_tool",
				Data: map[string]any{
					"records": []map[string]any{
						{"id": "1", "name": "Alice"},
					},
					"count": 1,
				},
			},
		},
		expected: []string{"Query executed", "records", "Alice"},
	}))

	t.Run("only tool results", runTest(struct {
		name      string
		response  *gollem.Response
		nextInput []gollem.Input
		expected  []string
	}{
		name:     "only tool results",
		response: &gollem.Response{},
		nextInput: []gollem.Input{
			gollem.FunctionResponse{
				ID:   "call_1",
				Name: "test_tool",
				Data: map[string]any{
					"data": "value",
				},
			},
		},
		expected: []string{"data", "value"},
	}))

	t.Run("nil response", runTest(struct {
		name      string
		response  *gollem.Response
		nextInput []gollem.Input
		expected  []string
	}{
		name:      "nil response",
		response:  nil,
		nextInput: []gollem.Input{},
		expected:  []string{},
	}))

	t.Run("with function calls", runTest(struct {
		name      string
		response  *gollem.Response
		nextInput []gollem.Input
		expected  []string
	}{
		name: "with function calls",
		response: &gollem.Response{
			FunctionCalls: []*gollem.FunctionCall{
				{Name: "query_tool", Arguments: map[string]any{"query": "SELECT *"}},
			},
		},
		nextInput: []gollem.Input{},
		expected:  []string{"Tool calls executed", "query_tool"},
	}))

	t.Run("mixed content", runTest(struct {
		name      string
		response  *gollem.Response
		nextInput []gollem.Input
		expected  []string
	}{
		name: "mixed content",
		response: &gollem.Response{
			Texts: []string{"Executed query"},
			FunctionCalls: []*gollem.FunctionCall{
				{Name: "query_tool"},
			},
		},
		nextInput: []gollem.Input{
			gollem.FunctionResponse{
				ID:   "call_1",
				Name: "query_tool",
				Data: map[string]any{"result": "success"},
			},
		},
		expected: []string{"Executed query", "Tool calls executed", "query_tool", "result", "success"},
	}))
}

// parseTaskResult writes its output into Task.Result, which buildPlanPrompt and
// buildReflectPrompt embed verbatim. Ranging over FunctionCall.Arguments without sorting
// made the same tool call render in a different order on every run.
func TestParseTaskResultOrdersArgumentsByName(t *testing.T) {
	response := &gollem.Response{
		FunctionCalls: []*gollem.FunctionCall{
			{
				ID:   "call_1",
				Name: "query_tool",
				Arguments: map[string]any{
					"zeta":  1,
					"alpha": 2,
					"mike":  3,
					"bravo": 4,
					"yank":  5,
				},
			},
		},
	}

	first := planexec.ParseTaskResult(response, nil)
	for range 50 {
		gt.Equal(t, first, planexec.ParseTaskResult(response, nil))
	}

	gt.S(t, first).Contains("  alpha: 2\n\n  bravo: 4\n\n  mike: 3\n\n  yank: 5\n\n  zeta: 1")
}
