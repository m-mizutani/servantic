package otel_test

import (
	"context"
	"errors"
	"testing"

	"github.com/gollem-dev/gollem/trace"
	traceOtel "github.com/gollem-dev/gollem/trace/otel"
	"github.com/m-mizutani/gt"
	sdkTrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func setupTestHandler() (trace.Handler, *tracetest.InMemoryExporter) {
	exporter := tracetest.NewInMemoryExporter()
	tp := sdkTrace.NewTracerProvider(
		sdkTrace.WithSyncer(exporter),
	)
	h := traceOtel.New(traceOtel.WithTracerProvider(tp))
	return h, exporter
}

func TestOTelHandlerImplementsHandler(t *testing.T) {
	h, _ := setupTestHandler()
	// Verify it implements trace.Handler
	_ = trace.Handler(h)
}

func TestOTelHandlerAgentExecute(t *testing.T) {
	h, exporter := setupTestHandler()
	ctx := context.Background()

	ctx = h.StartAgentExecute(ctx)
	h.EndAgentExecute(ctx, nil)

	spans := exporter.GetSpans()
	gt.Equal(t, len(spans), 1)
	gt.Equal(t, spans[0].Name, "agent_execute")
}

func TestOTelHandlerAgentExecuteWithError(t *testing.T) {
	h, exporter := setupTestHandler()
	ctx := context.Background()

	ctx = h.StartAgentExecute(ctx)
	h.EndAgentExecute(ctx, errors.New("test error"))

	spans := exporter.GetSpans()
	gt.Equal(t, len(spans), 1)
	gt.Equal(t, len(spans[0].Events), 1) // error event recorded
}

func TestOTelHandlerLLMCall(t *testing.T) {
	h, exporter := setupTestHandler()
	ctx := context.Background()

	ctx = h.StartAgentExecute(ctx)
	llmCtx := h.StartLLMCall(ctx)
	h.EndLLMCall(llmCtx, &trace.LLMCallData{
		Model:        "test-model",
		InputTokens:  100,
		OutputTokens: 50,
	}, nil)
	h.EndAgentExecute(ctx, nil)

	spans := exporter.GetSpans()
	gt.Equal(t, len(spans), 2) // llm_call + agent_execute

	// Find the llm_call span
	var llmSpan *tracetest.SpanStub
	for i := range spans {
		if spans[i].Name == "llm_call" {
			llmSpan = &spans[i]
			break
		}
	}
	gt.Value(t, llmSpan).NotNil()
}

func TestOTelHandlerToolExec(t *testing.T) {
	h, exporter := setupTestHandler()
	ctx := context.Background()

	ctx = h.StartAgentExecute(ctx)
	toolCtx := h.StartToolExec(ctx, "search", map[string]any{"query": "test"})
	h.EndToolExec(toolCtx, map[string]any{"found": true}, nil)
	h.EndAgentExecute(ctx, nil)

	spans := exporter.GetSpans()
	gt.Equal(t, len(spans), 2)

	var toolSpan *tracetest.SpanStub
	for i := range spans {
		if spans[i].Name == "tool:search" {
			toolSpan = &spans[i]
			break
		}
	}
	gt.Value(t, toolSpan).NotNil()
}

func TestOTelHandlerSubAgent(t *testing.T) {
	h, exporter := setupTestHandler()
	ctx := context.Background()

	ctx = h.StartAgentExecute(ctx)
	subCtx := h.StartSubAgent(ctx, "child")
	h.EndSubAgent(subCtx, nil)
	h.EndAgentExecute(ctx, nil)

	spans := exporter.GetSpans()
	gt.Equal(t, len(spans), 2)

	var subSpan *tracetest.SpanStub
	for i := range spans {
		if spans[i].Name == "sub_agent:child" {
			subSpan = &spans[i]
			break
		}
	}
	gt.Value(t, subSpan).NotNil()
}

func TestOTelHandlerAddEvent(t *testing.T) {
	h, exporter := setupTestHandler()
	ctx := context.Background()

	ctx = h.StartAgentExecute(ctx)

	type testData struct {
		Goal string `json:"goal"`
	}
	h.AddEvent(ctx, "plan_created", &testData{Goal: "test"})
	h.EndAgentExecute(ctx, nil)

	spans := exporter.GetSpans()
	gt.Equal(t, len(spans), 1)
	// agent_execute span should have one event
	gt.Equal(t, len(spans[0].Events), 1)
	gt.Equal(t, spans[0].Events[0].Name, "plan_created")
}

func TestOTelHandlerAddEventNilData(t *testing.T) {
	h, exporter := setupTestHandler()
	ctx := context.Background()

	ctx = h.StartAgentExecute(ctx)
	h.AddEvent(ctx, "some_event", nil)
	h.EndAgentExecute(ctx, nil)

	spans := exporter.GetSpans()
	gt.Equal(t, len(spans), 1)
	gt.Equal(t, len(spans[0].Events), 1)
	gt.Equal(t, spans[0].Events[0].Name, "some_event")
}

func TestOTelHandlerFinish(t *testing.T) {
	h, _ := setupTestHandler()
	// Finish should be a no-op and return nil
	err := h.Finish(context.Background())
	gt.NoError(t, err)
}

func TestOTelHandlerParentChildRelation(t *testing.T) {
	h, exporter := setupTestHandler()
	ctx := context.Background()

	agentCtx := h.StartAgentExecute(ctx)
	llmCtx := h.StartLLMCall(agentCtx)
	h.EndLLMCall(llmCtx, nil, nil)
	toolCtx := h.StartToolExec(agentCtx, "search", nil)
	h.EndToolExec(toolCtx, nil, nil)
	h.EndAgentExecute(agentCtx, nil)

	spans := exporter.GetSpans()
	gt.Equal(t, len(spans), 3)

	// Find agent span
	var agentSpan *tracetest.SpanStub
	for i := range spans {
		if spans[i].Name == "agent_execute" {
			agentSpan = &spans[i]
			break
		}
	}
	gt.Value(t, agentSpan).NotNil()

	// Verify child spans have agent as parent
	for i := range spans {
		if spans[i].Name != "agent_execute" {
			gt.Equal(t, spans[i].Parent.SpanID(), agentSpan.SpanContext.SpanID())
		}
	}
}

func TestOTelHandlerLLMCallCacheTokens(t *testing.T) {
	hasAttr := func(span *tracetest.SpanStub, key string) bool {
		for _, kv := range span.Attributes {
			if string(kv.Key) == key {
				return true
			}
		}
		return false
	}
	findLLMSpan := func(spans tracetest.SpanStubs) *tracetest.SpanStub {
		for i := range spans {
			if spans[i].Name == "llm_call" {
				return &spans[i]
			}
		}
		return nil
	}

	t.Run("records cache attributes when caching occurred", func(t *testing.T) {
		h, exporter := setupTestHandler()
		ctx := h.StartAgentExecute(context.Background())
		llmCtx := h.StartLLMCall(ctx)
		h.EndLLMCall(llmCtx, &trace.LLMCallData{
			Model:                    "m",
			InputTokens:              150,
			OutputTokens:             50,
			CacheCreationInputTokens: 10,
			CacheReadInputTokens:     100,
		}, nil)
		h.EndAgentExecute(ctx, nil)

		llmSpan := findLLMSpan(exporter.GetSpans())
		gt.Value(t, llmSpan).NotNil()
		gt.True(t, hasAttr(llmSpan, "llm.cache_creation_input_tokens"))
		gt.True(t, hasAttr(llmSpan, "llm.cache_read_input_tokens"))
	})

	t.Run("omits cache attributes when zero", func(t *testing.T) {
		h, exporter := setupTestHandler()
		ctx := h.StartAgentExecute(context.Background())
		llmCtx := h.StartLLMCall(ctx)
		h.EndLLMCall(llmCtx, &trace.LLMCallData{Model: "m", InputTokens: 100, OutputTokens: 50}, nil)
		h.EndAgentExecute(ctx, nil)

		llmSpan := findLLMSpan(exporter.GetSpans())
		gt.Value(t, llmSpan).NotNil()
		gt.False(t, hasAttr(llmSpan, "llm.cache_creation_input_tokens"))
		gt.False(t, hasAttr(llmSpan, "llm.cache_read_input_tokens"))
	})
}
