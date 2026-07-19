// Package otel provides an OpenTelemetry trace handler for gollem.
//
// It bridges gollem's trace events to OpenTelemetry spans, allowing
// integration with any OTel-compatible backend (Jaeger, Zipkin, OTLP, etc.).
//
// Basic usage with global TracerProvider:
//
//	agent := gollem.New(client, gollem.WithTrace(otel.New()))
//
// With explicit TracerProvider:
//
//	agent := gollem.New(client, gollem.WithTrace(
//	    otel.New(otel.WithTracerProvider(tp)),
//	))
package otel

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/gollem-dev/gollem/trace"
	otelAPI "go.opentelemetry.io/otel"
	otelTrace "go.opentelemetry.io/otel/trace"
)

const (
	tracerName = "github.com/gollem-dev/gollem"
)

// Option is a functional option for configuring the OTel handler.
type Option func(*handler)

// WithTracerProvider sets an explicit TracerProvider.
// If not set, the global TracerProvider is used.
func WithTracerProvider(tp otelTrace.TracerProvider) Option {
	return func(h *handler) {
		h.tracerProvider = tp
	}
}

// handler implements trace.Handler by bridging events to OpenTelemetry spans.
type handler struct {
	tracerProvider otelTrace.TracerProvider
	tracer         otelTrace.Tracer
}

// New creates a new OTel trace handler.
// If no TracerProvider is specified via options, the global TracerProvider is used.
func New(opts ...Option) trace.Handler {
	h := &handler{}
	for _, opt := range opts {
		opt(h)
	}

	if h.tracerProvider == nil {
		h.tracerProvider = otelAPI.GetTracerProvider()
	}
	h.tracer = h.tracerProvider.Tracer(tracerName)

	return h
}

func (h *handler) StartAgentExecute(ctx context.Context) context.Context {
	ctx, _ = h.tracer.Start(ctx, "agent_execute",
		otelTrace.WithSpanKind(otelTrace.SpanKindInternal),
	)
	return ctx
}

func (h *handler) EndAgentExecute(ctx context.Context, err error) {
	span := otelTrace.SpanFromContext(ctx)
	if err != nil {
		span.RecordError(err)
	}
	span.End()
}

func (h *handler) StartLLMCall(ctx context.Context) context.Context {
	ctx, _ = h.tracer.Start(ctx, "llm_call",
		otelTrace.WithSpanKind(otelTrace.SpanKindClient),
	)
	return ctx
}

func (h *handler) EndLLMCall(ctx context.Context, data *trace.LLMCallData, err error) {
	span := otelTrace.SpanFromContext(ctx)
	if data != nil {
		span.SetAttributes(
			llmModelAttr(data.Model),
			llmInputTokensAttr(data.InputTokens),
			llmOutputTokensAttr(data.OutputTokens),
		)
		// Record cache token breakdown only when caching occurred.
		if data.CacheCreationInputTokens > 0 {
			span.SetAttributes(llmCacheCreationInputTokensAttr(data.CacheCreationInputTokens))
		}
		if data.CacheReadInputTokens > 0 {
			span.SetAttributes(llmCacheReadInputTokensAttr(data.CacheReadInputTokens))
		}
	}
	if err != nil {
		span.RecordError(err)
	}
	span.End()
}

func (h *handler) StartToolExec(ctx context.Context, toolName string, args map[string]any) context.Context {
	ctx, _ = h.tracer.Start(ctx, fmt.Sprintf("tool:%s", toolName),
		otelTrace.WithSpanKind(otelTrace.SpanKindInternal),
	)
	span := otelTrace.SpanFromContext(ctx)
	span.SetAttributes(toolNameAttr(toolName))
	if args != nil {
		if b, err := json.Marshal(args); err == nil {
			span.SetAttributes(toolArgsAttr(string(b)))
		}
	}
	return ctx
}

func (h *handler) EndToolExec(ctx context.Context, result map[string]any, err error) {
	span := otelTrace.SpanFromContext(ctx)
	if err != nil {
		span.RecordError(err)
	}
	span.End()
}

func (h *handler) StartSubAgent(ctx context.Context, name string) context.Context {
	return h.startAgentSpan(ctx, "sub_agent", name)
}

func (h *handler) EndSubAgent(ctx context.Context, err error) {
	h.endSpan(ctx, err)
}

func (h *handler) StartChildAgent(ctx context.Context, name string) context.Context {
	return h.startAgentSpan(ctx, "child_agent", name)
}

func (h *handler) EndChildAgent(ctx context.Context, err error) {
	h.endSpan(ctx, err)
}

func (h *handler) startAgentSpan(ctx context.Context, prefix, name string) context.Context {
	ctx, _ = h.tracer.Start(ctx, fmt.Sprintf("%s:%s", prefix, name),
		otelTrace.WithSpanKind(otelTrace.SpanKindInternal),
	)
	return ctx
}

func (h *handler) endSpan(ctx context.Context, err error) {
	span := otelTrace.SpanFromContext(ctx)
	if err != nil {
		span.RecordError(err)
	}
	span.End()
}

func (h *handler) AddEvent(ctx context.Context, kind string, data any) {
	span := otelTrace.SpanFromContext(ctx)
	if data != nil {
		if b, err := json.Marshal(data); err == nil {
			span.AddEvent(kind, otelTrace.WithAttributes(eventDataAttr(string(b))))
		} else {
			span.AddEvent(kind)
		}
	} else {
		span.AddEvent(kind)
	}
}

func (h *handler) Finish(_ context.Context) error {
	// OTel spans are exported by the TracerProvider's SpanProcessor.
	// No additional finalization is needed here.
	return nil
}
