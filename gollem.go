package gollem

import (
	"context"
	"encoding/json"
	"log/slog"
	"slices"
	"time"

	"github.com/gollem-dev/gollem/trace"
	"github.com/google/uuid"
	"github.com/m-mizutani/goerr/v2"
)

// ResponseMode is the type for the response mode of the gollem agent.
type ResponseMode string

const (
	// ResponseModeBlocking is the response mode that blocks the prompt until the LLM generates a response. The agent will wait until all responses are ready.
	ResponseModeBlocking ResponseMode = "blocking"

	// ResponseModeStreaming is the response mode that streams the response from the LLM. The agent receives responses token by token.
	ResponseModeStreaming ResponseMode = "streaming"
)

func (x ResponseMode) String() string {
	return string(x)
}

// Agent is core structure of the package.
// Note: Agent is not thread-safe. Each instance should be used by a single goroutine
// or proper synchronization must be implemented by the caller.
type Agent struct {
	llm LLMClient

	gollemConfig

	// currentSession holds the current session for continuous execution
	// This field should only be accessed through session management methods
	// WARNING: Direct access is not thread-safe
	currentSession Session
}

// Session returns the current session for the agent.
// This is the only way to access the session and its history.
// If no session exists, this will return nil.
func (x *Agent) Session() Session {
	return x.currentSession
}

const (
	DefaultLoopLimit = 128
)

type gollemConfig struct {
	loopLimit    int
	systemPrompt string

	tools    []Tool
	toolSets []ToolSet

	responseMode ResponseMode
	logger       *slog.Logger
	history      *History
	strategy     Strategy

	// Content type and response schema for agent-level configuration
	contentType    ContentType
	responseSchema *Parameter

	// promptCache enables provider prompt caching for sessions the agent creates.
	promptCache bool

	// Middleware for content generation
	contentBlockMiddlewares  []ContentBlockMiddleware
	contentStreamMiddlewares []ContentStreamMiddleware

	// Middleware for tool execution
	toolMiddlewares []ToolMiddleware

	// Trace handler for agent execution tracing
	traceHandler trace.Handler

	// disableArgsValidation disables automatic argument validation before tool execution
	disableArgsValidation bool

	// historyRepo and historySessionID enable automatic history persistence.
	// When set, the agent loads history on first Execute and saves after each LLM round-trip.
	historyRepo      HistoryRepository
	historySessionID string
}

func (c *gollemConfig) Clone() *gollemConfig {
	return &gollemConfig{
		loopLimit:    c.loopLimit,
		systemPrompt: c.systemPrompt,

		tools:    c.tools[:],
		toolSets: c.toolSets[:],

		responseMode: c.responseMode,
		logger:       c.logger,

		history:  c.history,
		strategy: c.strategy,

		contentType:    c.contentType,
		responseSchema: c.responseSchema,
		promptCache:    c.promptCache,

		contentBlockMiddlewares:  c.contentBlockMiddlewares[:],
		contentStreamMiddlewares: c.contentStreamMiddlewares[:],
		toolMiddlewares:          c.toolMiddlewares[:],
		traceHandler:             c.traceHandler,

		disableArgsValidation: c.disableArgsValidation,

		historyRepo:      c.historyRepo,
		historySessionID: c.historySessionID,
	}
}

// New creates a new gollem agent.
func New(llmClient LLMClient, options ...Option) *Agent {
	s := &Agent{
		llm: llmClient,
		gollemConfig: gollemConfig{
			loopLimit:    DefaultLoopLimit,
			systemPrompt: "",

			responseMode: ResponseModeBlocking,
			logger:       slog.New(slog.DiscardHandler),
			strategy:     newDefaultStrategy(),
		},
	}

	for _, opt := range options {
		opt(&s.gollemConfig)
	}

	s.logger.Info("gollem agent created",
		"loop_limit", s.loopLimit,
		"system_prompt", s.systemPrompt,
		"tools_count", len(s.tools),
		"tool_sets_count", len(s.toolSets),
		"response_mode", s.responseMode,
		"has_history", s.history != nil,
	)

	return s
}

// Option is the type for the options of the gollem agent.
type Option func(*gollemConfig)

// WithLoopLimit sets the maximum number of loops for the gollem session iteration (ask LLM and execute tools is one loop).
func WithLoopLimit(loopLimit int) Option {
	return func(s *gollemConfig) {
		s.loopLimit = loopLimit
	}
}

// WithSystemPrompt sets the system prompt for the gollem agent. Default is no system prompt.
func WithSystemPrompt(systemPrompt string) Option {
	return func(s *gollemConfig) {
		s.systemPrompt = systemPrompt
	}
}

// WithTools sets the tools for the gollem agent.
func WithTools(tools ...Tool) Option {
	return func(s *gollemConfig) {
		s.tools = append(s.tools, tools...)
	}
}

// WithPromptCache enables provider prompt caching for sessions created by the
// agent. Currently only the Claude provider acts on it (ephemeral cache_control
// on the stable prefix and the growing conversation tail); OpenAI and Gemini
// cache automatically and are unaffected. Default: disabled.
func WithPromptCache(enabled bool) Option {
	return func(s *gollemConfig) {
		s.promptCache = enabled
	}
}

// WithToolSets sets the tool sets for the gollem agent.
func WithToolSets(toolSets ...ToolSet) Option {
	return func(s *gollemConfig) {
		s.toolSets = append(s.toolSets, toolSets...)
	}
}

// WithResponseMode sets the response mode for the gollem agent. Default is ResponseModeBlocking.
func WithResponseMode(responseMode ResponseMode) Option {
	return func(s *gollemConfig) {
		s.responseMode = responseMode
	}
}

// WithLogger sets the logger for the gollem agent. Default is discard logger.
func WithLogger(logger *slog.Logger) Option {
	return func(s *gollemConfig) {
		s.logger = logger
	}
}

// WithHistory sets the history for the gollem agent.
func WithHistory(history *History) Option {
	return func(s *gollemConfig) {
		s.history = history
	}
}

// WithContentBlockMiddleware adds a content block middleware to the agent.
// The middleware will be applied to all sessions created by this agent.
func WithContentBlockMiddleware(middleware ContentBlockMiddleware) Option {
	return func(s *gollemConfig) {
		s.contentBlockMiddlewares = append(s.contentBlockMiddlewares, middleware)
	}
}

// WithContentStreamMiddleware adds a content stream middleware to the agent.
// The middleware will be applied to all streaming sessions created by this agent.
func WithContentStreamMiddleware(middleware ContentStreamMiddleware) Option {
	return func(s *gollemConfig) {
		s.contentStreamMiddlewares = append(s.contentStreamMiddlewares, middleware)
	}
}

// WithToolMiddleware adds a tool middleware to the agent.
// The middleware will be applied to all tool executions by this agent.
func WithToolMiddleware(middleware ToolMiddleware) Option {
	return func(s *gollemConfig) {
		s.toolMiddlewares = append(s.toolMiddlewares, middleware)
	}
}

// WithSubAgents adds subagents to the agent.
// Subagents are converted to tools and can be invoked by the LLM.
// Each SubAgent implements the Tool interface, so they are added to the tools list.
func WithSubAgents(subagents ...*SubAgent) Option {
	return func(s *gollemConfig) {
		for _, subagent := range subagents {
			s.tools = append(s.tools, subagent)
		}
	}
}

// WithStrategy sets the strategy for execution. Default is SimpleLoop.
func WithStrategy(strategy Strategy) Option {
	return func(s *gollemConfig) {
		s.strategy = strategy
	}
}

// WithContentType sets the content type for the agent.
// This will be applied to all sessions created by this agent.
func WithContentType(contentType ContentType) Option {
	return func(s *gollemConfig) {
		s.contentType = contentType
	}
}

// WithResponseSchema sets the response schema for the agent.
// This will be applied to all sessions created by this agent.
// This option should be used with WithContentType(ContentTypeJSON).
func WithResponseSchema(schema *Parameter) Option {
	return func(s *gollemConfig) {
		s.responseSchema = schema
	}
}

// WithTrace sets the trace handler for the agent.
// When set, the agent will record execution traces including LLM calls,
// tool executions, and sub-agent invocations.
func WithTrace(handler trace.Handler) Option {
	return func(s *gollemConfig) {
		s.traceHandler = handler
	}
}

// WithHistoryRepository sets a HistoryRepository for automatic history persistence.
// When set, the agent automatically loads history on session creation and saves
// history after each LLM round-trip.
// The sessionID uniquely identifies the conversation session in the repository.
func WithHistoryRepository(repo HistoryRepository, sessionID string) Option {
	return func(s *gollemConfig) {
		s.historyRepo = repo
		s.historySessionID = sessionID
	}
}

// WithDisableArgsValidation disables automatic argument validation before tool execution.
// By default, the agent validates tool arguments from LLM against the tool's parameter
// specifications before calling Tool.Run(). Use this option to skip that validation.
func WithDisableArgsValidation() Option {
	return func(s *gollemConfig) {
		s.disableArgsValidation = true
	}
}

func setupTools(ctx context.Context, cfg *gollemConfig) (map[string]Tool, []Tool, error) {
	allTools := cfg.tools[:]

	toolMap, err := buildToolMap(ctx, allTools, cfg.toolSets)
	if err != nil {
		return nil, nil, err
	}

	// Order the tools by name instead of by map iteration order. The tool
	// definitions are the first element of the request prefix that Anthropic
	// matches the prompt cache against, so a list that is shuffled on every
	// execution invalidates the cache for every session.
	toolNames := make([]string, 0, len(toolMap))
	for name := range toolMap {
		toolNames = append(toolNames, name)
	}
	slices.Sort(toolNames)

	toolList := make([]Tool, 0, len(toolMap))
	for _, name := range toolNames {
		toolList = append(toolList, toolMap[name])
	}
	cfg.logger.Debug("gollem tool list", "names", toolNames)

	return toolMap, toolList, nil
}

// Execute performs the agent task with the given prompt. This method manages the session state internally,
// allowing for continuous conversation without manual history management.
// Returns (*ExecuteResponse, error) where ExecuteResponse contains the final conclusion.
// Use this method instead of Prompt for better agent-like behavior.
func (g *Agent) Execute(ctx context.Context, input ...Input) (_ *ExecuteResponse, err error) {
	cfg := g.Clone()
	logger := cfg.logger.With("gollem.exec_id", uuid.New().String())
	cfg.logger = logger

	logger.Debug("[start] gollem execution",
		"input", input,
		"has_existing_session", g.currentSession != nil,
	)
	defer logger.Debug("[end] gollem execution")

	// Setup trace handler if configured
	th := cfg.traceHandler
	if th != nil {
		ctx = trace.WithHandler(ctx, th)
		ctx = th.StartAgentExecute(ctx)
		defer func() {
			th.EndAgentExecute(ctx, err)
			if finishErr := th.Finish(ctx); finishErr != nil {
				logger.Warn("failed to finish trace", "error", finishErr)
			}
		}()
	}

	// Initialize strategy
	if err := cfg.strategy.Init(ctx, input); err != nil {
		return nil, goerr.Wrap(err, "failed to initialize strategy")
	}

	// Setup tools for the current execution
	toolMap, toolList, err := setupTools(ctx, cfg)
	if err != nil {
		return nil, err
	}

	// Get strategy-specific tools and merge them
	strategyTools, err := cfg.strategy.Tools(ctx)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to get strategy tools")
	}

	// Add strategy tools to the tool list
	for _, tool := range strategyTools {
		if _, ok := toolMap[tool.Spec().Name]; ok {
			return nil, goerr.Wrap(ErrToolNameConflict, "tool name conflict with strategy tool", goerr.V("tool_name", tool.Spec().Name))
		}
		toolList = append(toolList, tool)
		toolMap[tool.Spec().Name] = tool
	}

	// If no current session exists, create a new one
	if g.currentSession == nil {
		// WithHistory and WithHistoryRepository cannot be used together
		if cfg.history != nil && cfg.historyRepo != nil {
			return nil, goerr.New("WithHistory and WithHistoryRepository cannot be used together")
		}

		sessionOptions := []SessionOption{
			WithSessionSystemPrompt(cfg.systemPrompt),
		}

		// Add ContentType if specified
		if cfg.contentType != "" {
			sessionOptions = append(sessionOptions, WithSessionContentType(cfg.contentType))
		}

		// Add ResponseSchema if specified
		if cfg.responseSchema != nil {
			sessionOptions = append(sessionOptions, WithSessionResponseSchema(cfg.responseSchema))
		}

		// Propagate prompt-cache enablement to the session
		if cfg.promptCache {
			sessionOptions = append(sessionOptions, WithSessionPromptCache(true))
		}

		if cfg.history != nil {
			sessionOptions = append(sessionOptions, WithSessionHistory(cfg.history))
		}

		// Load history from repository if configured
		if cfg.historyRepo != nil {
			repoHistory, err := cfg.historyRepo.Load(ctx, cfg.historySessionID)
			if err != nil {
				return nil, goerr.Wrap(err, "failed to load history from repository",
					goerr.V("session_id", cfg.historySessionID))
			}
			if repoHistory != nil {
				sessionOptions = append(sessionOptions, WithSessionHistory(repoHistory))
			}
		}
		if len(toolList) > 0 {
			sessionOptions = append(sessionOptions, WithSessionTools(toolList...))
		}

		// Add middleware from agent configuration
		for _, mw := range cfg.contentBlockMiddlewares {
			sessionOptions = append(sessionOptions, WithSessionContentBlockMiddleware(mw))
		}
		for _, mw := range cfg.contentStreamMiddlewares {
			sessionOptions = append(sessionOptions, WithSessionContentStreamMiddleware(mw))
		}

		ssn, err := g.llm.NewSession(ctx, sessionOptions...)
		if err != nil {
			return nil, err
		}
		if ssn == nil {
			return nil, goerr.New("LLMClient.NewSession returned nil session")
		}
		g.currentSession = ssn
	}

	strategy := g.strategy

	var lastResponse *Response
	nextInput := input
	for i := 0; i < cfg.loopLimit; i++ {
		state := &StrategyState{
			Session:      g.currentSession,
			InitInput:    input,
			LastResponse: lastResponse,
			NextInput:    nextInput,
			Iteration:    i,
			Tools:        toolList,
			SystemPrompt: cfg.systemPrompt,
			History:      cfg.history.Clone(),
		}
		strategyInputs, executeResponse, err := strategy.Handle(ctx, state)
		if err != nil {
			return nil, err
		}

		logger.Debug("gollem input", "input", strategyInputs, "loop", i, "response", executeResponse)

		// ExecuteResponse priority processing
		if executeResponse != nil {
			// Input also specified? Log warning
			if len(strategyInputs) > 0 {
				logger.Warn("Strategy returned both ExecuteResponse and Input - Input will be ignored",
					"inputs_count", len(strategyInputs),
					"texts_count", len(executeResponse.Texts))
			}

			// Append user inputs to session history first
			// This is necessary when strategy returns ExecuteResponse without calling GenerateContent
			if len(executeResponse.UserInputs) > 0 {
				userHistory, err := convertInputsToHistory(executeResponse.UserInputs)
				if err != nil {
					return nil, goerr.Wrap(err, "failed to convert user inputs to history")
				}
				if userHistory != nil {
					if err := g.currentSession.AppendHistory(userHistory); err != nil {
						return nil, goerr.Wrap(err, "failed to append user inputs to session history")
					}
					if err := saveHistoryToRepo(ctx, g.currentSession, cfg); err != nil {
						return nil, err
					}
				}
			}

			// Append final response texts to session history as assistant message
			if len(executeResponse.Texts) > 0 {
				// Combine all texts into a single message
				var combinedText string
				for i, text := range executeResponse.Texts {
					if i > 0 {
						combinedText += "\n"
					}
					combinedText += text
				}

				textData, err := json.Marshal(map[string]string{"text": combinedText})
				if err != nil {
					return nil, goerr.Wrap(err, "failed to marshal text content")
				}

				// Create a history entry for the texts (as assistant message)
				textHistory := &History{
					Version: HistoryVersion,
					Messages: []Message{
						{
							Role: RoleAssistant,
							Contents: []MessageContent{
								{
									Type: MessageContentTypeText,
									Data: textData,
								},
							},
						},
					},
				}
				if err := g.currentSession.AppendHistory(textHistory); err != nil {
					return nil, goerr.Wrap(err, "failed to append texts to session history")
				}
				if err := saveHistoryToRepo(ctx, g.currentSession, cfg); err != nil {
					return nil, err
				}
			}

			// Return strategy's response immediately
			return executeResponse, nil
		}

		// Input processing
		if len(strategyInputs) == 0 {
			// Both nil: session terminated
			return nil, nil
		}

		switch cfg.responseMode {
		case ResponseModeBlocking:
			output, err := g.currentSession.Generate(ctx, strategyInputs)
			if err != nil {
				return nil, err
			}

			newInput, err := handleResponse(ctx, logger, output, toolMap, cfg.toolMiddlewares, cfg.disableArgsValidation)
			if err != nil {
				return nil, err
			}
			if err := saveHistoryToRepo(ctx, g.currentSession, cfg); err != nil {
				return nil, err
			}
			lastResponse = output
			nextInput = newInput

		case ResponseModeStreaming:
			stream, err := g.currentSession.Stream(ctx, strategyInputs)
			if err != nil {
				return nil, err
			}
			nextInput = []Input{}

			// Accumulate the complete response for lastResponse
			var streamedResponse Response
			for output := range stream {
				logger.Debug("recv response", "output", output)
				newInput, err := handleResponse(ctx, logger, output, toolMap, cfg.toolMiddlewares, cfg.disableArgsValidation)
				if err != nil {
					return nil, err
				}
				nextInput = append(nextInput, newInput...)

				// Accumulate streaming response.
				// Token usage is a per-call snapshot repeated across chunks (each
				// chunk carries the running/full total, not a delta), so take the
				// latest non-zero value rather than summing to avoid multiplying
				// the usage by the number of chunks.
				streamedResponse.Texts = append(streamedResponse.Texts, output.Texts...)
				streamedResponse.FunctionCalls = append(streamedResponse.FunctionCalls, output.FunctionCalls...)
				if output.InputToken > 0 {
					streamedResponse.InputToken = output.InputToken
				}
				if output.OutputToken > 0 {
					streamedResponse.OutputToken = output.OutputToken
				}
				if output.CacheCreationInputToken > 0 {
					streamedResponse.CacheCreationInputToken = output.CacheCreationInputToken
				}
				if output.CacheReadInputToken > 0 {
					streamedResponse.CacheReadInputToken = output.CacheReadInputToken
				}
				if output.Error != nil {
					streamedResponse.Error = output.Error
				}
			}
			if err := saveHistoryToRepo(ctx, g.currentSession, cfg); err != nil {
				return nil, err
			}
			lastResponse = &streamedResponse
		}
	}

	return nil, goerr.Wrap(ErrLoopLimitExceeded, "session stopped", goerr.V("loop_limit", cfg.loopLimit))
}

// saveHistoryToRepo saves the current session history to the configured HistoryRepository.
// It is a no-op if no repository is configured.
func saveHistoryToRepo(ctx context.Context, session Session, cfg *gollemConfig) error {
	if cfg.historyRepo == nil {
		return nil
	}
	history, err := session.History()
	if err != nil {
		return goerr.Wrap(err, "failed to get session history for save")
	}
	if err := cfg.historyRepo.Save(ctx, cfg.historySessionID, history); err != nil {
		return goerr.Wrap(err, "failed to save history to repository",
			goerr.V("session_id", cfg.historySessionID))
	}
	return nil
}

func handleResponse(ctx context.Context, logger *slog.Logger, output *Response, toolMap map[string]Tool, toolMiddlewares []ToolMiddleware, disableArgsValidation bool) ([]Input, error) {

	newInput := make([]Input, 0)

	logger.Debug("[start] handling response", "function_calls", output.FunctionCalls)
	defer logger.Debug("[exit] handling response")

	// Call the ToolRequestHook for all tool calls
	for _, toolCall := range output.FunctionCalls {
		logger = logger.With("call", toolCall)

		tool, ok := toolMap[toolCall.Name]
		if !ok {
			logger.Info("gollem tool not found")
			newInput = append(newInput, FunctionResponse{
				Name:  toolCall.Name,
				ID:    toolCall.ID,
				Error: goerr.New(toolCall.Name+" is not found", goerr.V("call", toolCall)),
			})
			continue
		}

		resp, err := executeToolCall(ctx, logger, toolCall, tool, toolMiddlewares, disableArgsValidation)
		if err != nil {
			return nil, err
		}
		newInput = append(newInput, resp)
	}

	return newInput, nil
}

// executeToolCall executes a single tool call with trace span management via defer.
func executeToolCall(ctx context.Context, logger *slog.Logger, toolCall *FunctionCall, tool Tool, toolMiddlewares []ToolMiddleware, disableArgsValidation bool) (_ FunctionResponse, retErr error) {

	// Start tool execution trace span
	var toolResult map[string]any
	if h := trace.HandlerFrom(ctx); h != nil {
		ctx = h.StartToolExec(ctx, toolCall.Name, toolCall.Arguments)
		defer func() { h.EndToolExec(ctx, toolResult, retErr) }()
	}

	// Create base tool handler
	baseHandler := func(ctx context.Context, req *ToolExecRequest) (*ToolExecResponse, error) {
		// Validate arguments before execution
		if !disableArgsValidation && req.ToolSpec != nil {
			if err := req.ToolSpec.ValidateArgs(req.Tool.Arguments); err != nil {
				return &ToolExecResponse{
					Error: err,
				}, nil
			}
		}

		start := time.Now()
		result, err := tool.Run(ctx, req.Tool.Arguments)
		duration := time.Since(start).Milliseconds()

		return &ToolExecResponse{
			Result:   result,
			Error:    err,
			Duration: duration,
		}, nil
	}

	// Build middleware chain
	handler := buildToolChain(toolMiddlewares, baseHandler)

	// Execute tool with middleware
	toolSpec := tool.Spec()
	req := &ToolExecRequest{
		Tool:     toolCall,
		ToolSpec: &toolSpec,
	}

	resp, err := handler(ctx, req)
	if err != nil {
		logger.Info("gollem tool handler error", "error", err)
		return FunctionResponse{
			ID:    toolCall.ID,
			Name:  toolCall.Name,
			Error: goerr.With(err, goerr.V("call", toolCall)),
		}, nil
	}

	toolResult = resp.Result
	if resp.Error != nil {
		retErr = resp.Error
		logger.Info("gollem tool error", "error", resp.Error)
		return FunctionResponse{
			ID:    toolCall.ID,
			Name:  toolCall.Name,
			Error: goerr.With(resp.Error, goerr.V("call", toolCall)),
		}, nil
	}

	logger.Debug("gollem tool result", "tool", toolCall.Name, "result", toolResult, "duration_ms", resp.Duration)

	// Sanitize result to ensure a generic JSON-compatible structure for LLM processing.
	if toolResult != nil {
		marshaled, err := json.Marshal(toolResult)
		if err != nil {
			return FunctionResponse{}, goerr.Wrap(err, "failed to marshal result", goerr.V("result", toolResult))
		}
		var unmarshaled map[string]any
		if err := json.Unmarshal(marshaled, &unmarshaled); err != nil {
			return FunctionResponse{}, goerr.Wrap(err, "failed to unmarshal result", goerr.V("marshaled", string(marshaled)))
		}
		toolResult = unmarshaled
	}

	return FunctionResponse{
		ID:   toolCall.ID,
		Name: toolCall.Name,
		Data: toolResult,
	}, nil
}

type toolWrapper struct {
	spec ToolSpec
	run  func(ctx context.Context, args map[string]any) (map[string]any, error)
}

func (x *toolWrapper) Spec() ToolSpec {
	return x.spec
}

func (x *toolWrapper) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return x.run(ctx, args)
}

func buildToolMap(ctx context.Context, tools []Tool, toolSets []ToolSet) (map[string]Tool, error) {
	toolMap := map[string]Tool{}

	for _, tool := range tools {
		if _, ok := toolMap[tool.Spec().Name]; ok {
			return nil, goerr.Wrap(ErrToolNameConflict, "tool name conflict (builtin tools)", goerr.V("tool_name", tool.Spec().Name))
		}
		toolMap[tool.Spec().Name] = tool
	}

	for _, toolSet := range toolSets {
		specs, err := toolSet.Specs(ctx)
		if err != nil {
			return nil, goerr.Wrap(err, "failed to get tool set specs")
		}

		for _, spec := range specs {
			if _, ok := toolMap[spec.Name]; ok {
				return nil, goerr.Wrap(ErrToolNameConflict, "tool name conflict (builtin tool sets)", goerr.V("tool_name", spec.Name))
			}
			toolMap[spec.Name] = &toolWrapper{
				spec: spec,
				run: func(ctx context.Context, args map[string]any) (map[string]any, error) {
					return toolSet.Run(ctx, spec.Name, args)
				},
			}
		}
	}

	return toolMap, nil
}

// convertInputsToHistory converts a slice of Input to History with user role
func convertInputsToHistory(inputs []Input) (*History, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	var contents []MessageContent

	for _, input := range inputs {
		switch v := input.(type) {
		case Text:
			textData, err := json.Marshal(map[string]string{"text": string(v)})
			if err != nil {
				return nil, goerr.Wrap(err, "failed to marshal text content")
			}
			contents = append(contents, MessageContent{
				Type: MessageContentTypeText,
				Data: textData,
			})

		case Image:
			imageData, err := json.Marshal(map[string]string{
				"data":      v.Base64(),
				"mime_type": v.MimeType(),
			})
			if err != nil {
				return nil, goerr.Wrap(err, "failed to marshal image content")
			}
			contents = append(contents, MessageContent{
				Type: MessageContentTypeImage,
				Data: imageData,
			})

		case PDF:
			mc, err := NewPDFContent(v.Data(), "")
			if err != nil {
				return nil, goerr.Wrap(err, "failed to marshal PDF content")
			}
			contents = append(contents, mc)

		case FunctionResponse:
			// FunctionResponse is not user input, skip it
			// It should be handled separately in the normal flow
			continue

		default:
			return nil, goerr.New("unsupported input type for user history")
		}
	}

	if len(contents) == 0 {
		return nil, nil
	}

	return &History{
		Version: HistoryVersion,
		Messages: []Message{
			{
				Role:     RoleUser,
				Contents: contents,
			},
		},
	}, nil
}
