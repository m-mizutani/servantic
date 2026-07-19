package logger

import (
	"context"
	"log/slog"
	"time"

	"github.com/gollem-dev/gollem/trace"
)

// Event represents a trace event type that can be selectively enabled.
type Event int

const (
	// AgentExec enables logging of agent execution start/end.
	AgentExec Event = iota
	// LLMRequest enables logging of LLM request details (system prompt, messages, tools).
	LLMRequest
	// LLMResponse enables logging of LLM response details (texts, function calls, token usage).
	LLMResponse
	// ToolExec enables logging of tool execution (name, args, result, duration).
	ToolExec
	// SubAgent enables logging of sub-agent start/end.
	SubAgent
	// ChildAgent enables logging of child agent start/end.
	ChildAgent
	// CustomEvent enables logging of strategy-defined custom events.
	CustomEvent

	eventCount // sentinel for iteration
)

type config struct {
	logger *slog.Logger
	events map[Event]bool
}

// Option configures the logger handler.
type Option func(*config)

// WithLogger sets a custom slog.Logger. Default is a discard logger.
func WithLogger(l *slog.Logger) Option {
	return func(c *config) {
		c.logger = l
	}
}

// WithEvents enables only the specified event types.
// When not specified, all events are enabled.
func WithEvents(events ...Event) Option {
	return func(c *config) {
		c.events = make(map[Event]bool, len(events))
		for _, e := range events {
			c.events[e] = true
		}
	}
}

// handler implements trace.Handler by logging events via slog.
type handler struct {
	cfg config
}

// New creates a new trace.Handler that logs trace events via slog.
// By default, all events are enabled. Use WithEvents to enable only specific events.
func New(opts ...Option) trace.Handler {
	cfg := config{}
	for _, opt := range opts {
		opt(&cfg)
	}

	// Default: all events enabled
	if cfg.events == nil {
		cfg.events = make(map[Event]bool, eventCount)
		for i := Event(0); i < eventCount; i++ {
			cfg.events[i] = true
		}
	}

	return &handler{cfg: cfg}
}

func (h *handler) logger() *slog.Logger {
	if h.cfg.logger != nil {
		return h.cfg.logger
	}
	return slog.New(slog.DiscardHandler)
}

func (h *handler) enabled(e Event) bool {
	return h.cfg.events[e]
}

// context key for storing span start time
type startTimeKey struct{}

func withStartTime(ctx context.Context, t time.Time) context.Context {
	return context.WithValue(ctx, startTimeKey{}, t)
}

func startTimeFrom(ctx context.Context) time.Time {
	t, _ := ctx.Value(startTimeKey{}).(time.Time)
	return t
}

// context key for storing tool call info
type toolInfoKey struct{}

type toolInfo struct {
	name string
	args map[string]any
}

func withToolInfo(ctx context.Context, info toolInfo) context.Context {
	return context.WithValue(ctx, toolInfoKey{}, info)
}

func toolInfoFrom(ctx context.Context) toolInfo {
	info, _ := ctx.Value(toolInfoKey{}).(toolInfo)
	return info
}

// context key for storing sub-agent name
type subAgentNameKey struct{}

func withSubAgentName(ctx context.Context, name string) context.Context {
	return context.WithValue(ctx, subAgentNameKey{}, name)
}

func subAgentNameFrom(ctx context.Context) string {
	name, _ := ctx.Value(subAgentNameKey{}).(string)
	return name
}

// StartAgentExecute logs agent execution start.
func (h *handler) StartAgentExecute(ctx context.Context) context.Context {
	if h.enabled(AgentExec) {
		h.logger().DebugContext(ctx, "agent execution started")
	}
	return withStartTime(ctx, time.Now())
}

// EndAgentExecute logs agent execution end with duration and error info.
func (h *handler) EndAgentExecute(ctx context.Context, err error) {
	if !h.enabled(AgentExec) {
		return
	}

	attrs := []any{
		slog.Duration("duration", time.Since(startTimeFrom(ctx))),
	}
	if err != nil {
		attrs = append(attrs, slog.String("error", err.Error()))
	}
	h.logger().DebugContext(ctx, "agent execution ended", attrs...)
}

// StartLLMCall records the start time for duration calculation.
func (h *handler) StartLLMCall(ctx context.Context) context.Context {
	return withStartTime(ctx, time.Now())
}

// EndLLMCall logs LLM call details based on enabled events.
// LLMRequest controls request details, LLMResponse controls response details.
// If either is enabled, model and token usage are always included.
func (h *handler) EndLLMCall(ctx context.Context, data *trace.LLMCallData, err error) {
	reqEnabled := h.enabled(LLMRequest)
	respEnabled := h.enabled(LLMResponse)
	if !reqEnabled && !respEnabled {
		return
	}

	attrs := []any{
		slog.Duration("duration", time.Since(startTimeFrom(ctx))),
	}

	if data != nil {
		attrs = append(attrs,
			slog.String("model", data.Model),
			slog.Int("input_tokens", data.InputTokens),
			slog.Int("output_tokens", data.OutputTokens),
		)
		// Emit cache token breakdown only when caching occurred, to avoid noise.
		if data.CacheCreationInputTokens > 0 {
			attrs = append(attrs, slog.Int("cache_creation_input_tokens", data.CacheCreationInputTokens))
		}
		if data.CacheReadInputTokens > 0 {
			attrs = append(attrs, slog.Int("cache_read_input_tokens", data.CacheReadInputTokens))
		}

		if reqEnabled && data.Request != nil {
			attrs = append(attrs, slog.Any("request", data.Request))
		}
		if respEnabled && data.Response != nil {
			attrs = append(attrs, slog.Any("response", data.Response))
		}
	}

	if err != nil {
		attrs = append(attrs, slog.String("error", err.Error()))
	}

	h.logger().DebugContext(ctx, "llm call", attrs...)
}

// StartToolExec records the start time and tool info for EndToolExec.
func (h *handler) StartToolExec(ctx context.Context, toolName string, args map[string]any) context.Context {
	ctx = withStartTime(ctx, time.Now())
	ctx = withToolInfo(ctx, toolInfo{name: toolName, args: args})
	return ctx
}

// EndToolExec logs tool execution details.
func (h *handler) EndToolExec(ctx context.Context, result map[string]any, err error) {
	if !h.enabled(ToolExec) {
		return
	}

	info := toolInfoFrom(ctx)
	attrs := []any{
		slog.String("tool", info.name),
		slog.Any("args", info.args),
		slog.Duration("duration", time.Since(startTimeFrom(ctx))),
		slog.Any("result", result),
	}
	if err != nil {
		attrs = append(attrs, slog.String("error", err.Error()))
	}
	h.logger().DebugContext(ctx, "tool execution", attrs...)
}

// StartSubAgent logs sub-agent start and stores the name in context.
func (h *handler) StartSubAgent(ctx context.Context, name string) context.Context {
	ctx = withStartTime(ctx, time.Now())
	ctx = withSubAgentName(ctx, name)
	if h.enabled(SubAgent) {
		h.logger().DebugContext(ctx, "sub agent started", slog.String("name", name))
	}
	return ctx
}

// EndSubAgent logs sub-agent end with duration and error info.
func (h *handler) EndSubAgent(ctx context.Context, err error) {
	if !h.enabled(SubAgent) {
		return
	}

	attrs := []any{
		slog.String("name", subAgentNameFrom(ctx)),
		slog.Duration("duration", time.Since(startTimeFrom(ctx))),
	}
	if err != nil {
		attrs = append(attrs, slog.String("error", err.Error()))
	}
	h.logger().DebugContext(ctx, "sub agent ended", attrs...)
}

// StartChildAgent logs child agent start and stores the name in context.
func (h *handler) StartChildAgent(ctx context.Context, name string) context.Context {
	ctx = withStartTime(ctx, time.Now())
	ctx = withSubAgentName(ctx, name)
	if h.enabled(ChildAgent) {
		h.logger().DebugContext(ctx, "child agent started", slog.String("name", name))
	}
	return ctx
}

// EndChildAgent logs child agent end with duration and error info.
func (h *handler) EndChildAgent(ctx context.Context, err error) {
	if !h.enabled(ChildAgent) {
		return
	}

	attrs := []any{
		slog.String("name", subAgentNameFrom(ctx)),
		slog.Duration("duration", time.Since(startTimeFrom(ctx))),
	}
	if err != nil {
		attrs = append(attrs, slog.String("error", err.Error()))
	}
	h.logger().DebugContext(ctx, "child agent ended", attrs...)
}

// AddEvent logs a custom strategy event.
func (h *handler) AddEvent(ctx context.Context, kind string, data any) {
	if !h.enabled(CustomEvent) {
		return
	}

	h.logger().DebugContext(ctx, "event",
		slog.String("kind", kind),
		slog.Any("data", data),
	)
}

// Finish is a no-op for the logger handler. Persistence is the Recorder's responsibility.
func (h *handler) Finish(_ context.Context) error {
	return nil
}
