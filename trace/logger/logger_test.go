package logger_test

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"testing"

	"github.com/gollem-dev/gollem/trace"
	"github.com/gollem-dev/gollem/trace/logger"
	"github.com/m-mizutani/gt"
)

// logEntry captures a single slog record for testing.
type logEntry struct {
	Level   slog.Level
	Message string
	Attrs   map[string]any
}

// testHandler is a slog.Handler that captures log records for assertions.
type testHandler struct {
	mu      sync.Mutex
	entries []logEntry
}

func (h *testHandler) Enabled(_ context.Context, _ slog.Level) bool { return true }
func (h *testHandler) WithAttrs(_ []slog.Attr) slog.Handler         { return h }
func (h *testHandler) WithGroup(_ string) slog.Handler              { return h }
func (h *testHandler) Handle(_ context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	attrs := make(map[string]any)
	r.Attrs(func(a slog.Attr) bool {
		attrs[a.Key] = a.Value.Any()
		return true
	})

	h.entries = append(h.entries, logEntry{
		Level:   r.Level,
		Message: r.Message,
		Attrs:   attrs,
	})
	return nil
}

func (h *testHandler) getEntries() []logEntry {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]logEntry, len(h.entries))
	copy(out, h.entries)
	return out
}

func newTestLogger() (*slog.Logger, *testHandler) {
	th := &testHandler{}
	return slog.New(th), th
}

func TestAgentExecuteLogging(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(logger.WithLogger(slogger))
	ctx := context.Background()

	agentCtx := h.StartAgentExecute(ctx)
	h.EndAgentExecute(agentCtx, nil)

	entries := th.getEntries()
	gt.Equal(t, len(entries), 2)
	gt.Equal(t, entries[0].Message, "agent execution started")
	gt.Equal(t, entries[1].Message, "agent execution ended")
	gt.Value(t, entries[1].Attrs["duration"]).NotNil()
}

func TestAgentExecuteWithError(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(logger.WithLogger(slogger))
	ctx := context.Background()

	agentCtx := h.StartAgentExecute(ctx)
	h.EndAgentExecute(agentCtx, errors.New("something went wrong"))

	entries := th.getEntries()
	gt.Equal(t, len(entries), 2)
	gt.Equal(t, entries[1].Attrs["error"], "something went wrong")
}

func TestLLMCallLogging(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(logger.WithLogger(slogger))
	ctx := context.Background()

	llmCtx := h.StartLLMCall(ctx)
	h.EndLLMCall(llmCtx, &trace.LLMCallData{
		InputTokens:  100,
		OutputTokens: 50,
		Model:        "test-model",
		Request: &trace.LLMRequest{
			SystemPrompt: "You are helpful",
			Messages:     []trace.Message{{Role: "user", Contents: []trace.MessageContent{trace.NewTextContent("hello")}}},
		},
		Response: &trace.LLMResponse{
			Texts: []string{"Hi there!"},
		},
	}, nil)

	entries := th.getEntries()
	gt.Equal(t, len(entries), 1)
	gt.Equal(t, entries[0].Message, "llm call")
	gt.Equal(t, entries[0].Attrs["model"], "test-model")
	gt.Value(t, entries[0].Attrs["input_tokens"]).NotNil()
	gt.Value(t, entries[0].Attrs["output_tokens"]).NotNil()
	gt.Value(t, entries[0].Attrs["request"]).NotNil()
	gt.Value(t, entries[0].Attrs["response"]).NotNil()
}

func TestLLMCallWithError(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(logger.WithLogger(slogger))
	ctx := context.Background()

	llmCtx := h.StartLLMCall(ctx)
	h.EndLLMCall(llmCtx, nil, errors.New("llm error"))

	entries := th.getEntries()
	gt.Equal(t, len(entries), 1)
	gt.Equal(t, entries[0].Attrs["error"], "llm error")
}

func TestLLMCallNilData(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(logger.WithLogger(slogger))
	ctx := context.Background()

	llmCtx := h.StartLLMCall(ctx)
	h.EndLLMCall(llmCtx, nil, nil)

	entries := th.getEntries()
	// Should still produce a log entry (LLMRequest and LLMResponse both enabled by default)
	gt.Equal(t, len(entries), 1)
	gt.Equal(t, entries[0].Message, "llm call")
}

func TestToolExecLogging(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(logger.WithLogger(slogger))
	ctx := context.Background()

	toolCtx := h.StartToolExec(ctx, "search", map[string]any{"query": "test"})
	h.EndToolExec(toolCtx, map[string]any{"found": true}, nil)

	entries := th.getEntries()
	gt.Equal(t, len(entries), 1)
	gt.Equal(t, entries[0].Message, "tool execution")
	gt.Equal(t, entries[0].Attrs["tool"], "search")
	gt.Value(t, entries[0].Attrs["args"]).NotNil()
	gt.Value(t, entries[0].Attrs["result"]).NotNil()
	gt.Value(t, entries[0].Attrs["duration"]).NotNil()
}

func TestToolExecWithError(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(logger.WithLogger(slogger))
	ctx := context.Background()

	toolCtx := h.StartToolExec(ctx, "search", nil)
	h.EndToolExec(toolCtx, nil, errors.New("tool failed"))

	entries := th.getEntries()
	gt.Equal(t, len(entries), 1)
	gt.Equal(t, entries[0].Attrs["error"], "tool failed")
}

func TestSubAgentLogging(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(logger.WithLogger(slogger))
	ctx := context.Background()

	subCtx := h.StartSubAgent(ctx, "planner")
	h.EndSubAgent(subCtx, nil)

	entries := th.getEntries()
	gt.Equal(t, len(entries), 2)
	gt.Equal(t, entries[0].Message, "sub agent started")
	gt.Equal(t, entries[0].Attrs["name"], "planner")
	gt.Equal(t, entries[1].Message, "sub agent ended")
	gt.Equal(t, entries[1].Attrs["name"], "planner")
	gt.Value(t, entries[1].Attrs["duration"]).NotNil()
}

func TestSubAgentWithError(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(logger.WithLogger(slogger))
	ctx := context.Background()

	subCtx := h.StartSubAgent(ctx, "executor")
	h.EndSubAgent(subCtx, errors.New("sub agent error"))

	entries := th.getEntries()
	gt.Equal(t, len(entries), 2)
	gt.Equal(t, entries[1].Attrs["error"], "sub agent error")
}

func TestCustomEventLogging(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(logger.WithLogger(slogger))
	ctx := context.Background()

	type planData struct {
		Goal string `json:"goal"`
	}
	h.AddEvent(ctx, "plan_created", &planData{Goal: "implement feature"})

	entries := th.getEntries()
	gt.Equal(t, len(entries), 1)
	gt.Equal(t, entries[0].Message, "event")
	gt.Equal(t, entries[0].Attrs["kind"], "plan_created")
	gt.Value(t, entries[0].Attrs["data"]).NotNil()
}

func TestWithEventsFilterLLMRequestOnly(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(
		logger.WithLogger(slogger),
		logger.WithEvents(logger.LLMRequest),
	)
	ctx := context.Background()

	// LLM call: should log with request but without response
	llmCtx := h.StartLLMCall(ctx)
	h.EndLLMCall(llmCtx, &trace.LLMCallData{
		Model:        "test-model",
		InputTokens:  100,
		OutputTokens: 50,
		Request: &trace.LLMRequest{
			SystemPrompt: "prompt",
		},
		Response: &trace.LLMResponse{
			Texts: []string{"response text"},
		},
	}, nil)

	entries := th.getEntries()
	gt.Equal(t, len(entries), 1)
	gt.Equal(t, entries[0].Message, "llm call")
	gt.Value(t, entries[0].Attrs["request"]).NotNil()
	// response should not be present
	_, hasResponse := entries[0].Attrs["response"]
	gt.B(t, hasResponse).False()
}

func TestWithEventsFilterLLMResponseOnly(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(
		logger.WithLogger(slogger),
		logger.WithEvents(logger.LLMResponse),
	)
	ctx := context.Background()

	llmCtx := h.StartLLMCall(ctx)
	h.EndLLMCall(llmCtx, &trace.LLMCallData{
		Model:        "test-model",
		InputTokens:  100,
		OutputTokens: 50,
		Request: &trace.LLMRequest{
			SystemPrompt: "prompt",
		},
		Response: &trace.LLMResponse{
			Texts: []string{"response text"},
		},
	}, nil)

	entries := th.getEntries()
	gt.Equal(t, len(entries), 1)
	gt.Value(t, entries[0].Attrs["response"]).NotNil()
	// request should not be present
	_, hasRequest := entries[0].Attrs["request"]
	gt.B(t, hasRequest).False()
}

func TestWithEventsDisablesNonSelected(t *testing.T) {
	slogger, th := newTestLogger()
	// Only enable ToolExec
	h := logger.New(
		logger.WithLogger(slogger),
		logger.WithEvents(logger.ToolExec),
	)
	ctx := context.Background()

	// Agent exec: should NOT log
	agentCtx := h.StartAgentExecute(ctx)
	h.EndAgentExecute(agentCtx, nil)

	// LLM call: should NOT log
	llmCtx := h.StartLLMCall(ctx)
	h.EndLLMCall(llmCtx, &trace.LLMCallData{Model: "test"}, nil)

	// Tool exec: SHOULD log
	toolCtx := h.StartToolExec(ctx, "search", nil)
	h.EndToolExec(toolCtx, nil, nil)

	// Sub agent: should NOT log
	subCtx := h.StartSubAgent(ctx, "child")
	h.EndSubAgent(subCtx, nil)

	// Custom event: should NOT log
	h.AddEvent(ctx, "test_event", nil)

	entries := th.getEntries()
	gt.Equal(t, len(entries), 1)
	gt.Equal(t, entries[0].Message, "tool execution")
}

func TestWithEventsNeitherLLMRequestNorResponse(t *testing.T) {
	slogger, th := newTestLogger()
	// Only enable AgentExec (neither LLMRequest nor LLMResponse)
	h := logger.New(
		logger.WithLogger(slogger),
		logger.WithEvents(logger.AgentExec),
	)
	ctx := context.Background()

	llmCtx := h.StartLLMCall(ctx)
	h.EndLLMCall(llmCtx, &trace.LLMCallData{Model: "test"}, nil)

	entries := th.getEntries()
	gt.Equal(t, len(entries), 0)
}

func TestFinishReturnsNil(t *testing.T) {
	h := logger.New()
	err := h.Finish(context.Background())
	gt.NoError(t, err)
}

func TestDefaultLogger(t *testing.T) {
	// When no WithLogger is specified, should not panic
	h := logger.New()
	ctx := context.Background()

	agentCtx := h.StartAgentExecute(ctx)
	h.EndAgentExecute(agentCtx, nil)
	// No assertion needed; just verify it doesn't panic
}

func TestWithMultipleEvents(t *testing.T) {
	slogger, th := newTestLogger()
	h := logger.New(
		logger.WithLogger(slogger),
		logger.WithEvents(logger.AgentExec, logger.ToolExec, logger.CustomEvent),
	)
	ctx := context.Background()

	agentCtx := h.StartAgentExecute(ctx)
	toolCtx := h.StartToolExec(ctx, "search", nil)
	h.EndToolExec(toolCtx, nil, nil)
	h.AddEvent(ctx, "plan", nil)
	h.EndAgentExecute(agentCtx, nil)

	entries := th.getEntries()
	gt.Equal(t, len(entries), 4)
	gt.Equal(t, entries[0].Message, "agent execution started")
	gt.Equal(t, entries[1].Message, "tool execution")
	gt.Equal(t, entries[2].Message, "event")
	gt.Equal(t, entries[3].Message, "agent execution ended")
}

func TestLLMCallCacheTokenLogging(t *testing.T) {
	t.Run("emits cache tokens when caching occurred", func(t *testing.T) {
		slogger, th := newTestLogger()
		h := logger.New(logger.WithLogger(slogger))
		ctx := context.Background()

		llmCtx := h.StartLLMCall(ctx)
		h.EndLLMCall(llmCtx, &trace.LLMCallData{
			InputTokens:              150,
			OutputTokens:             50,
			Model:                    "test-model",
			CacheCreationInputTokens: 10,
			CacheReadInputTokens:     100,
		}, nil)

		entries := th.getEntries()
		gt.Equal(t, len(entries), 1)
		gt.Value(t, entries[0].Attrs["cache_creation_input_tokens"]).NotNil()
		gt.Value(t, entries[0].Attrs["cache_read_input_tokens"]).NotNil()
	})

	t.Run("omits cache tokens when zero", func(t *testing.T) {
		slogger, th := newTestLogger()
		h := logger.New(logger.WithLogger(slogger))
		ctx := context.Background()

		llmCtx := h.StartLLMCall(ctx)
		h.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 100, OutputTokens: 50, Model: "m"}, nil)

		entries := th.getEntries()
		gt.Equal(t, len(entries), 1)
		_, hasCreation := entries[0].Attrs["cache_creation_input_tokens"]
		_, hasRead := entries[0].Attrs["cache_read_input_tokens"]
		gt.False(t, hasCreation)
		gt.False(t, hasRead)
	})
}
